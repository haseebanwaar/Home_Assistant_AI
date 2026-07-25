import base64
import io
import json
import logging
import os
import asyncio
import datetime
import time
import uuid
import wave
from threading import Lock

import nest_asyncio
from collections import deque
from dotenv import load_dotenv
import pydub
import soundfile as sf
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from lmdeploy.vl.utils import encode_image_base64
from starlette.responses import JSONResponse, StreamingResponse

from providers.asr.parakeet import nemo_transcribe, parakeet_health
from providers.local_openAI import client, get_model_name_vlm
from providers.tts.kokoro.kokoro_tts import run_kokoro
from vector_store.activity_logger import ActivityLogger
from sources.screen import RealtimeScreenCapture
from sources.camera_manager import CameraManager
from vector_store.rag.activity_retriever import ActivityRetriever
from memory.retrieval.evidence import EvidenceRetriever
from tools.registry import ToolRegistry, register_default_tools
from agents.proactive import ProactiveNarrator
from memory.notifications import NotificationCenter


load_dotenv()
nest_asyncio.apply()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("home_assistant")

# When true, per-stage debug/timing events are also streamed to the client as NDJSON.
def env_bool(name, default=False):
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")


def env_int(name, default, minimum=1):
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}, got {value}")
    return value


def env_float(name, default, minimum=0.01):
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}, got {value}")
    return value


DEBUG_VERBOSE = env_bool("DEBUG_VERBOSE")
MAX_FRAMES = env_int("MAX_FRAMES", 20)
MAX_MEMORY_ITEMS = env_int("MAX_MEMORY_ITEMS", 20)

app = FastAPI(title="Home Assistant AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === GLOBALS (single-user POC: one conversation, one active context) ===
_chat_history = []
_current_context = "talker"
vlm_model = None
screen_stream = None
camera_manager = None
camera_bootstrap_task = None
proactive = None
mobile_activity_task = None
daily_report_task = None  # nightly Coach report scheduler
memory_pipeline = None   # Step-a: live sessions/events/knowledge pipeline
neo4j_store = None       # optional graph sink for the live pipeline
_proactive_insights = deque(maxlen=20)
_proactive_seq = 0
notification_center = NotificationCenter(
    path=os.getenv("NOTIFICATION_STORE_PATH", "data/notifications.json"),
    important_cooldown_seconds=env_int(
        "NOTIFICATION_COOLDOWN_SECONDS", 600, minimum=0),
)


class MobileFrameStream:
    """Thread-safe buffer populated by the Flutter capture service."""
    def __init__(self, max_frames=120):
        self.frame_buffer = deque(maxlen=max_frames)
        self.lock = Lock()
        self.active = False
        self.source = None
        self.frames_received = 0
        self.last_frame_at = None
        self.last_error = None
        self.last_processed_at = None

    def start(self, source):
        with self.lock:
            self.frame_buffer.clear()
            self.active, self.source = True, source
            self.frames_received, self.last_frame_at, self.last_error = 0, None, None
            self.last_processed_at = None

    def stop(self):
        with self.lock:
            self.active = False

    def add(self, image):
        with self.lock:
            if not self.active:
                return False
            self.frame_buffer.append(image.copy())
            self.frames_received += 1
            self.last_frame_at = time.time()
            return True

    def frames(self, source):
        with self.lock:
            return list(self.frame_buffer) if self.active and self.source == source else []

    def processing_window(self):
        """Take the current window while retaining two frames for continuity."""
        with self.lock:
            if not self.active or len(self.frame_buffer) < 2:
                return self.source, []
            frames = list(self.frame_buffer)
            self.frame_buffer.clear()
            self.frame_buffer.extend(frames[-2:])
            return self.source, frames

    def processed(self):
        with self.lock:
            self.last_processed_at = time.time()

    def status(self):
        with self.lock:
            age = time.time() - self.last_frame_at if self.last_frame_at else None
            return {"configured": True, "active": self.active, "source": self.source,
                    "healthy": self.active and age is not None and age < 10,
                    "buffered_frames": len(self.frame_buffer), "frames_received": self.frames_received,
                    "last_frame_age_seconds": round(age, 1) if age is not None else None,
                    "last_processed_at": self.last_processed_at,
                    "error": self.last_error}


mobile_stream = MobileFrameStream()
_pipeline_status = {"active": False, "stage": "ready", "turn": None, "updated_at": time.time()}


def set_pipeline_status(active, stage, turn=None):
    _pipeline_status.update(active=active, stage=stage, turn=turn, updated_at=time.time())

# Create a single, shared Qdrant client instance
qdrant_client = QdrantClient(path=os.getenv("QDRANT_PATH", "./qdrant_db"))

past_memory = ActivityRetriever(client=qdrant_client)
activity_logger = ActivityLogger(client=qdrant_client)

# The single retrieval path shared by /memory/search, room chat and the grounded
# assistant, so all three answer the same question the same way. Reuses
# past_memory's cross-encoder instead of loading a second copy; its graph handle
# is attached at startup, once Neo4j is known to be up.
evidence_retriever = EvidenceRetriever(
    qdrant_client=qdrant_client, reranker=past_memory.reranker)

# Tools the assistant may call (function-calling, not MCP — see tools/registry.py).
tool_registry = ToolRegistry()
register_default_tools(tool_registry, past_memory)


INITIATIVE_PROMPT = """Act as an initiative-taking partner, not a passive question-answering \
system. Use your own judgment and creativity to notice implications, make connections, \
anticipate needs, and offer useful next steps or ideas the user did not explicitly request. \
Do not reduce your judgment to a fixed rubric of what is right or wrong. Stay grounded in \
the available context and clearly distinguish remembered facts from inference."""


CONCISE_SYSTEM_PROMPT = f"""You are a conversational AI designed for a real-time Speech-to-Speech (S2S) system. Your primary function is to engage in natural, fluid conversation.

    {INITIATIVE_PROMPT}

    Follow these critical rules:
    1.  **Be Concise:** Keep your responses short, typically one or two sentences. Avoid long paragraphs at all costs.
    2.  **Sound Natural:** Speak like a real person. Use contractions (e.g., "it's," "don't," "you're") and a friendly, conversational tone.
    3.  **TTS-Friendly:** Your responses will be spoken aloud by a Text-to-Speech (TTS) engine. Use simple sentence structures and common vocabulary that are easy to pronounce and sound natural when spoken.
    4.  **No Formatting:** Do not use lists, bullet points, markdown, or any text formatting. Your output is for voice only.

    Your goal is to keep the conversation moving, not to provide exhaustive, written-out answers.
    """

def validate_configuration():
    """Fail at startup with actionable messages for required POC settings."""
    vlm_url = os.getenv("VLM_BASE_URL", "http://0.0.0.0:8000/v1").strip()
    if not vlm_url.startswith(("http://", "https://")):
        raise RuntimeError("VLM_BASE_URL must start with http:// or https://")

    asr_url = os.getenv("PARAKEET_SERVER_URL", "http://127.0.0.1:8765").strip()
    if not asr_url.startswith(("http://", "https://")):
        raise RuntimeError("PARAKEET_SERVER_URL must start with http:// or https://")

    env_int("APP_PORT", 8000)
    env_int("MAX_FRAMES", 20)
    env_int("MAX_MEMORY_ITEMS", 20)
    if env_bool("SCREEN_CAPTURE_ENABLED", True):
        env_int("SCREEN_MONITOR_INDEX", 1)
        env_int("SCREEN_WINDOW_SECONDS", 60)
        env_float("SCREEN_FPS", 1.0)
    if env_bool("CAMERA_CAPTURE_ENABLED", True):
        env_int("CAMERA_WINDOW_SECONDS", 60)
        env_float("CAMERA_FPS", 1.0)


# === STARTUP ===
@app.on_event("startup")
async def startup_event():
    global vlm_model, screen_stream, camera_manager, camera_bootstrap_task
    global proactive, mobile_activity_task, daily_report_task
    global memory_pipeline, neo4j_store

    validate_configuration()
    logger.info("Loading model...")
    try:
        vlm_model = await get_model_name_vlm()
    except Exception as exc:
        logger.error(
            "VLM server unreachable at %s: %s",
            os.getenv("VLM_BASE_URL", "http://0.0.0.0:8000/v1"), exc,
        )
        raise
    logger.info("Model loaded: %s", vlm_model)

    # Optional proactive narrator: speaks unprompted insights about screen activity.
    insight_callback = None
    if env_bool("PROACTIVE_ENABLED", True):
        proactive = ProactiveNarrator(
            vlm_model, client,
            cooldown_seconds=env_int("PROACTIVE_COOLDOWN_SECONDS", 120),
            focus_cooldown_seconds=env_int("PROACTIVE_FOCUS_COOLDOWN_SECONDS", 60),
            retriever=evidence_retriever,
            # Lazy — the graph is connected further down in this same startup.
            store_getter=lambda: neo4j_store,
        )
        insight_callback = handle_screen_description
        logger.info("Proactive narrator enabled (cooldown=%ds).", proactive.cooldown_seconds)
    else:
        logger.info("Proactive narrator disabled.")

    # Step-a: optional live memory pipeline (sessions/events/knowledge + stores).
    # Fully opt-in — unset LIVE_MEMORY leaves the legacy per-minute path unchanged.
    if env_bool("LIVE_MEMORY", True):
        from memory.pipeline import MemoryPipeline
        if env_bool("MEMORY_NEO4J", True):
            try:
                from memory.stores.neo4j_store import Neo4jStore
                neo4j_store = Neo4jStore()
                neo4j_store.verify()
                neo4j_store.apply_schema()
                logger.info("LIVE_MEMORY: Neo4j graph sink enabled (%s).", neo4j_store.uri)
            except Exception as exc:
                logger.warning("MEMORY_NEO4J on but Neo4j unavailable (%s) — "
                               "continuing without the graph sink.", exc)
                neo4j_store = None
        memory_pipeline = MemoryPipeline(
            id_strategy="deterministic",
            expected_seconds=env_int("SCREEN_WINDOW_SECONDS", 60),
            neo4j_store=neo4j_store,
            activity_logger=activity_logger,  # event-scoped Qdrant sink
            jsonl=True,                        # keep /debug/timeline populated
            notification_sink=notification_center.consider_event,
        )
        logger.info("LIVE_MEMORY enabled (graph=%s).", neo4j_store is not None)
    else:
        logger.info("LIVE_MEMORY disabled (legacy per-minute logging).")

    evidence_retriever.neo4j = neo4j_store

    # The two capture rooms exist from boot, so the feed has somewhere to land on
    # the very first observation. Cameras is created by the camera manager.
    if neo4j_store is not None:
        try:
            neo4j_store.ensure_source_room("screen")
        except Exception as exc:
            logger.warning("ensure_source_room(screen) failed: %s", exc)

    # Give the assistant graph-backed memory tools when the graph is available.
    if neo4j_store is not None:
        from tools.graph_tools import register_graph_tools
        register_graph_tools(tool_registry, lambda: neo4j_store)
        logger.info("Registered graph memory tools: %s", tool_registry.names)

    if env_bool("SCREEN_CAPTURE_ENABLED", True):
        screen_stream = RealtimeScreenCapture(
            video_source="",
            model_name_vlm=vlm_model,
            window_size=env_int("SCREEN_WINDOW_SECONDS", 60),
            fps=env_float("SCREEN_FPS", 1.0),
            monitor_index=env_int("SCREEN_MONITOR_INDEX", 1),
            activity_logger=activity_logger,
            insight_callback=insight_callback,
            pipeline=memory_pipeline,
        )
        logger.info("Screen capture enabled (monitor=%d).", screen_stream.monitor_index)
    else:
        logger.info("Screen capture disabled.")

    # Cameras: discover live ONVIF cameras (+ any explicit RTSP URLs) and run a
    # worker per camera, each feeding its own room. Discovery does blocking
    # network I/O, so bootstrap it off the event loop and let cameras come online
    # shortly after startup rather than blocking the whole app on WS-Discovery.
    if env_bool("CAMERA_CAPTURE_ENABLED", True):
        camera_manager = CameraManager(
            model_name_vlm=vlm_model, neo4j_store=neo4j_store,
            activity_logger=activity_logger,
            window_seconds=env_int("CAMERA_WINDOW_SECONDS", 60),
            fps=env_float("CAMERA_FPS", 1.0),
            notification_sink=notification_center.consider_event,
            insight_callback=handle_observation_description if proactive is not None else None,
        )
        camera_bootstrap_task = asyncio.create_task(
            asyncio.to_thread(camera_manager.discover_and_start))
        logger.info("Camera discovery started in background.")
    else:
        logger.info("Camera capture disabled (CAMERA_CAPTURE_ENABLED=0).")

    logger.info("Single-user POC pipeline ready.")
    mobile_activity_task = asyncio.create_task(process_mobile_activity())

    if neo4j_store is not None and env_bool("DAILY_REPORT_SCHEDULED", True):
        daily_report_task = asyncio.create_task(run_daily_report_scheduler())
        logger.info("Daily report scheduled for %02d:%02d local.",
                    env_int("DAILY_REPORT_HOUR", 23, minimum=0),
                    env_int("DAILY_REPORT_MINUTE", 30, minimum=0))


async def run_daily_report_scheduler():
    """Generate and post the Coach's daily report each evening.

    The report existed but nothing ever called it, so the Daily room only
    updated when the user remembered to press a button — which makes a
    "lifelong memory" feel inert. Runs for the day that is ending.
    """
    hour = min(env_int("DAILY_REPORT_HOUR", 23, minimum=0), 23)
    minute = min(env_int("DAILY_REPORT_MINUTE", 30, minimum=0), 59)
    while True:
        now = datetime.datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += datetime.timedelta(days=1)
        try:
            await asyncio.sleep((target - now).total_seconds())
        except asyncio.CancelledError:
            raise
        try:
            result = await daily_report(date=target.date().isoformat(), post=True)
            logger.info("Scheduled daily report for %s (posted=%s).",
                        result.get("date"), result.get("posted"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never let one bad night kill the scheduler.
            logger.warning("Scheduled daily report failed: %s", exc)


@app.on_event("shutdown")
async def shutdown_event():
    for task in (mobile_activity_task, daily_report_task):
        if task is None:
            continue
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    if camera_bootstrap_task is not None:
        camera_bootstrap_task.cancel()
        try:
            await camera_bootstrap_task
        except (asyncio.CancelledError, Exception):
            pass
    if screen_stream is not None:
        screen_stream.cleanup()   # also flushes the live memory pipeline
    if camera_manager is not None:
        camera_manager.cleanup_all()
    if neo4j_store is not None:
        try:
            neo4j_store.close()
        except Exception:
            logger.warning("Neo4j store close failed", exc_info=True)


# === STATUS / DEBUG ENDPOINTS ===
@app.get("/ready")
async def ready():
    """Simple readiness probe — is the VLM model resolved?"""
    qdrant_ready = False
    try:
        qdrant_client.get_collections()
        qdrant_ready = True
    except Exception:
        logger.exception("Qdrant readiness check failed")
    asr = await asyncio.to_thread(parakeet_health)
    return {
        "ready": vlm_model is not None and qdrant_ready and asr["ready"],
        "vlm_model": vlm_model,
        "qdrant": qdrant_ready,
        "asr": asr,
    }


@app.get("/status")
async def status():
    """Runtime status for debugging (no sessions — single POC user)."""
    asr = await asyncio.to_thread(parakeet_health)
    return {
        "ready": vlm_model is not None and asr["ready"],
        "vlm_model": vlm_model,
        "asr": asr,
        "context": _current_context,
        "history_turns": len(_chat_history) // 2,
        "screen_stream": screen_stream.status() if screen_stream else {"configured": False},
        "screen_frames": len(screen_stream.frame_buffer) if screen_stream else 0,
        "cameras": camera_manager.status_all() if camera_manager else [],
        "mobile_capture": mobile_stream.status(),
        "pipeline": dict(_pipeline_status),
        "debug_verbose": DEBUG_VERBOSE,
    }


@app.post("/capture/control")
async def capture_control(request: Request):
    """Start or stop processing frames sent by the Flutter app."""
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid JSON: {exc}"})
    action = data.get("action") if isinstance(data, dict) else None
    source = data.get("source") if isinstance(data, dict) else None
    if action == "start" and source in ("camera", "screen"):
        mobile_stream.start(source)
        logger.info("Mobile %s processing started.", source)
    elif action == "stop":
        mobile_stream.stop()
        logger.info("Mobile capture processing stopped.")
    else:
        return JSONResponse(status_code=400, content={"error": "invalid capture action/source"})
    return mobile_stream.status()


@app.post("/screen/control")
async def screen_control(request: Request):
    """Pause or resume the desktop screen capture from the UI."""
    if screen_stream is None:
        return JSONResponse(status_code=400, content={"error": "screen capture disabled"})
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid JSON: {exc}"})
    action = data.get("action") if isinstance(data, dict) else None
    if action == "pause":
        screen_stream.pause()
    elif action == "resume":
        screen_stream.resume()
    else:
        return JSONResponse(status_code=400, content={"error": "action must be pause|resume"})
    return screen_stream.status()


@app.get("/cameras")
async def cameras_list():
    """Per-camera status (live health, paused, events logged)."""
    return {"cameras": camera_manager.status_all() if camera_manager else []}


@app.post("/cameras/{camera_id:path}/control")
async def camera_control(camera_id: str, request: Request):
    """Pause or resume a single camera worker."""
    if camera_manager is None:
        return JSONResponse(status_code=400, content={"error": "camera capture disabled"})
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid JSON: {exc}"})
    action = data.get("action") if isinstance(data, dict) else None
    if action == "pause":
        ok = camera_manager.pause(camera_id)
    elif action == "resume":
        ok = camera_manager.resume(camera_id)
    else:
        return JSONResponse(status_code=400, content={"error": "action must be pause|resume"})
    if not ok:
        return JSONResponse(status_code=404, content={"error": "camera not found"})
    worker = camera_manager.workers.get(camera_id)
    return worker.status() if worker else {"ok": True}


@app.post("/capture/frame")
async def capture_frame(request: Request):
    """Receive one JPEG frame from the Android foreground capture service."""
    if not mobile_stream.active:
        return JSONResponse(status_code=409, content={"error": "capture processing is stopped"})
    try:
        image = Image.open(io.BytesIO(await request.body())).convert("RGB")
        image.load()
    except Exception as exc:
        mobile_stream.last_error = str(exc)
        return JSONResponse(status_code=400, content={"error": f"invalid image: {exc}"})
    mobile_stream.add(image)
    return {"accepted": True, "frames_received": mobile_stream.frames_received}


@app.get("/debug/last-extraction")
async def last_extraction():
    """Step 2 debug handle: the most recent structured extraction record."""
    path = os.path.join(os.getenv("DEBUG_DIR", os.path.join("data", "debug")), "extractions.jsonl")
    if not os.path.exists(path):
        return {"extraction": None, "note": "no extractions yet (set STRUCTURED_EXTRACTION=1)"}
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return {"extraction": None}
    return {"extraction": json.loads(last)}


@app.get("/debug/timeline")
async def debug_timeline():
    """Step 6 debug handle: today's sessions -> events with spans."""
    import datetime

    debug_dir = os.getenv("DEBUG_DIR", os.path.join("data", "debug"))

    def _load(name):
        path = os.path.join(debug_dir, f"{name}.jsonl")
        out = []
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    sessions = _load("sessions")
    events = _load("events")

    today = datetime.date.today()

    def _is_today(ts):
        try:
            return datetime.date.fromtimestamp(ts) == today
        except (OverflowError, OSError, ValueError):
            return False

    by_session = {}
    for e in events:
        by_session.setdefault(e["session_id"], []).append(e)

    timeline = []
    for s in sessions:
        if not _is_today(s.get("start", 0)):
            continue
        evs = sorted(by_session.get(s["session_id"], []), key=lambda e: e["span_start"])
        timeline.append({
            "session_id": s["session_id"],
            "activity_type": s["activity_type"],
            "application": s["application"],
            "project_id": s.get("project_id"),
            "state": s.get("state"),
            "active_seconds": s.get("active_seconds"),
            "resume_count": s.get("resume_count", 0),
            "events": [
                {"event_id": e["event_id"], "span_start": e["span_start"],
                 "span_end": e["span_end"], "span_seconds": e["span_seconds"],
                 "boundary_label": e.get("boundary_label"), "summary": e.get("summary", "")[:80]}
                for e in evs
            ],
        })
    return {"date": today.isoformat(), "sessions": len(timeline), "timeline": timeline}


# === STABLE MEMORY API (consumed by the Flutter app) ===
def _today_iso():
    import datetime
    return datetime.date.today().isoformat()


@app.get("/rooms")
async def rooms_list(include_archived: bool = False):
    """List rooms (activity/project/topic/daily) with event counts."""
    if neo4j_store is None:
        return {"error": "graph not enabled (start with MEMORY_NEO4J=1)"}
    return {"rooms": neo4j_store.list_rooms(include_archived=include_archived)}


@app.post("/rooms")
async def rooms_create(request: Request):
    """Create a user-managed topic room with optional routing matchers."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("body must be an object")
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")
        from memory.models.room import Room, RoomMatcher
        from memory.rooms.registry import _slug
        room_id = str(data.get("room_id") or f"topic:{_slug(name)}").strip()
        matcher = RoomMatcher(**(data.get("matcher") or {}))
        room = Room(
            room_id=room_id, name=name, kind="topic", auto=False, matcher=matcher,
            description=str(data.get("description") or "").strip(),
            instructions=str(data.get("instructions") or "").strip(),
            color=str(data.get("color") or "#8B7CF6"),
            icon=str(data.get("icon") or "forum"),
            pinned=bool(data.get("pinned", False)),
            position=int(data.get("position") or 0),
        )
        created = neo4j_store.create_room(room)
        return JSONResponse(status_code=201, content={"room": created})
    except (TypeError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/rooms/hygiene")
async def rooms_hygiene(stale_days: int = None, thin_minutes: int = None):
    """What's cluttering the Rooms screen: stale auto rooms and likely duplicates.

    Proposals only — nothing is archived or merged without an explicit call.
    Registered BEFORE /rooms/{room_id} so it isn't swallowed by that route.
    """
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    from memory.rooms.hygiene import (
        DEFAULT_STALE_DAYS, DEFAULT_THIN_MINUTES, merge_suggestions, stale_rooms)

    stats = neo4j_store.room_stats()
    overlaps = neo4j_store.room_overlap()
    return {
        "rooms": len(stats),
        "stale": stale_rooms(
            stats,
            stale_days=stale_days if stale_days is not None else DEFAULT_STALE_DAYS,
            thin_minutes=(thin_minutes if thin_minutes is not None
                          else DEFAULT_THIN_MINUTES)),
        "merges": merge_suggestions(overlaps, stats),
    }


@app.get("/rooms/{room_id}")
async def room_get(room_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    room = neo4j_store.get_room(room_id)
    if room is None:
        return JSONResponse(status_code=404, content={"error": "room not found"})
    return {"room": room}


@app.patch("/rooms/{room_id}")
async def room_update(room_id: str, request: Request):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("body must be an object")
        if "name" in data and not str(data["name"]).strip():
            raise ValueError("name cannot be empty")
        if "matcher" in data:
            from memory.models.room import RoomMatcher
            data["matcher"] = RoomMatcher(**(data["matcher"] or {}))
        room = neo4j_store.update_room(room_id, data)
        if room is None:
            return JSONResponse(status_code=404, content={"error": "room not found"})
        return {"room": room}
    except (TypeError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.delete("/rooms/{room_id}")
async def room_delete(room_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        if not neo4j_store.delete_room(room_id):
            return JSONResponse(status_code=404, content={"error": "room not found"})
        return {"deleted": True, "room_id": room_id}
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/rooms/{room_id}/reroute")
async def room_reroute(room_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    if neo4j_store.get_room(room_id) is None:
        return JSONResponse(status_code=404, content={"error": "room not found"})
    return neo4j_store.reroute_events(room_id=room_id)


@app.get("/rooms/{room_id}/feed")
async def room_feed(room_id: str, date: str = None, limit: int = 200,
                    offset: int = 0, kinds: str = None, q: str = None,
                    priorities: str = None, flagged: bool = None):
    """A room's merged feed — events + user notes + chat, newest first."""
    if neo4j_store is None:
        return {"error": "graph not enabled (start with MEMORY_NEO4J=1)"}
    selected_kinds = [k.strip() for k in kinds.split(",") if k.strip()] if kinds else None
    selected_priorities = ([p.strip() for p in priorities.split(",") if p.strip()]
                           if priorities else None)
    return {"room_id": room_id, "date": date, "offset": offset, "limit": limit,
            "feed": neo4j_store.room_feed_full(
                room_id, date_str=date, limit=limit, offset=offset,
                kinds=selected_kinds, query=q,
                priorities=selected_priorities, flagged=flagged)}


@app.post("/rooms/daily/report")
async def daily_report(date: str = None, post: bool = True):
    """Phase 3: generate the Coach's productivity report for a day and (by
    default) post it into the Daily room."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    import datetime
    from memory.summary.coach import format_report, coach_prompt

    ds = date or _today_iso()
    metrics = neo4j_store.daily_metrics(ds)
    claims = neo4j_store.day_claims(ds, limit=8)
    entities = neo4j_store.day_entities(ds, limit=12)

    feedback = ""
    if metrics.get("events"):
        try:
            resp = await client.chat.completions.create(
                model=vlm_model,
                messages=[{"role": "user", "content": coach_prompt(metrics, claims)}],
                max_tokens=350)
            feedback = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("coach feedback LLM failed: %s", exc)

    report = format_report(metrics, claims=claims, entities=entities)
    if feedback:
        report += f"\n\n## Coach\n{feedback}"

    posted = bool(post and metrics.get("events"))
    if posted:
        eod = datetime.datetime.fromisoformat(ds).timestamp() + 86399
        neo4j_store.add_message("daily", "coach", report, ts=eod)

    return {"date": ds, "metrics": metrics, "feedback": feedback,
            "report": report, "posted": posted}


@app.post("/rooms/hygiene/archive")
async def rooms_hygiene_archive(request: Request):
    """Archive the given rooms (reversible — archived rooms can be restored)."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid JSON: {exc}"})
    room_ids = data.get("room_ids") if isinstance(data, dict) else None
    if not isinstance(room_ids, list) or not room_ids:
        return JSONResponse(status_code=400,
                            content={"error": "room_ids must be a non-empty list"})
    return {"archived": neo4j_store.archive_rooms(room_ids)}


@app.post("/rooms/consolidate")
async def rooms_consolidate(purge_empty: bool = True):
    """Fold legacy per-activity/per-project/per-camera rooms into Screen/Cameras.

    A one-shot migration for graphs written before rooms were per capture source.
    Events are relinked, manual assignments are respected, and legacy rooms that
    hold notes or chat are archived instead of deleted.
    """
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        return neo4j_store.consolidate_source_rooms(purge_empty=purge_empty)
    except Exception as exc:
        logger.exception("room consolidation failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/rooms/{room_id}/merge")
async def room_merge(room_id: str, request: Request):
    """Merge this room into another, then archive it."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid JSON: {exc}"})
    target = (data.get("target_room_id") or "").strip() if isinstance(data, dict) else ""
    if not target:
        return JSONResponse(status_code=400,
                            content={"error": "target_room_id is required"})
    try:
        merged = neo4j_store.merge_rooms(room_id, target)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if merged is None:
        return JSONResponse(status_code=404, content={"error": "room not found"})
    return {"merged": merged}


@app.post("/rooms/{room_id}/promote")
async def room_promote(room_id: str, request: Request):
    """Keep an auto room for good: makes it user-owned so hygiene ignores it."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        data = {}
    name = (data.get("name") or "").strip() or None if isinstance(data, dict) else None
    pinned = data.get("pinned") if isinstance(data, dict) else None
    promoted = neo4j_store.promote_room(room_id, name=name, pinned=pinned)
    if promoted is None:
        return JSONResponse(status_code=404, content={"error": "room not found"})
    return {"room": promoted}


@app.get("/rooms/{room_id}/arc")
async def room_arc(room_id: str, weeks: int = 8, narrate: bool = False):
    """How a room's work has gone week over week.

    Everything else in the API is day-scoped, which makes long-term memory
    invisible; this is the view that shows a project's actual arc.
    """
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    weeks = max(1, min(int(weeks), 52))
    today = datetime.date.today()
    # Start on the Monday `weeks` weeks back, so buckets align to whole weeks.
    this_monday = today - datetime.timedelta(days=today.weekday())
    first_monday = this_monday - datetime.timedelta(weeks=weeks - 1)
    start = datetime.datetime.combine(first_monday, datetime.time.min).timestamp()
    end = datetime.datetime.combine(
        this_monday + datetime.timedelta(days=7), datetime.time.min).timestamp()

    room = neo4j_store.get_room(room_id)
    if room is None:
        return JSONResponse(status_code=404, content={"error": "room not found"})

    buckets = neo4j_store.room_weekly(room_id, start, end)
    for bucket in buckets:
        week_start = datetime.date.fromisoformat(bucket["week_start"])
        week_from = datetime.datetime.combine(
            week_start, datetime.time.min).timestamp()
        week_to = week_from + 7 * 86400
        highlights = neo4j_store.room_week_highlights(room_id, week_from, week_to)
        bucket["claims"] = highlights["claims"]
        bucket["entities"] = neo4j_store.room_week_entities(
            room_id, week_from, week_to)

    total_minutes = sum(b.get("active_minutes") or 0 for b in buckets)
    result = {
        "room_id": room_id, "room": room.get("name"), "weeks": weeks,
        "buckets": buckets,
        "summary": {
            "active_minutes": total_minutes,
            "active_weeks": sum(1 for b in buckets if (b.get("events") or 0) > 0),
            "events": sum(b.get("events") or 0 for b in buckets),
        },
    }

    if narrate and buckets:
        lines = []
        for bucket in buckets:
            if not bucket.get("events"):
                continue
            entities = ", ".join(e["name"] for e in bucket.get("entities", [])[:5])
            lines.append(
                f"Week of {bucket['week_start']}: "
                f"{bucket['active_minutes']:.0f} min over {bucket['active_days']} days"
                + (f"; worked with {entities}" if entities else "")
                + ("; " + " ".join(bucket["claims"][:3]) if bucket.get("claims") else ""))
        prompt = (
            "You are summarizing how a user's project went over several weeks, "
            "from their own screen-activity records. Write 4-6 sentences describing "
            "the arc: what they started with, how the focus shifted, what got done, "
            "and where it stands now. Be specific and do not invent anything.\n\n"
            f"Project/room: {room.get('name')}\n\n" + "\n".join(lines))
        try:
            response = await client.chat.completions.create(
                model=vlm_model, messages=[{"role": "user", "content": prompt}],
                max_tokens=400)
            result["narrative"] = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("room arc narration failed: %s", exc)
            result["narrative"] = None

    return result


@app.post("/rooms/{room_id}/note")
async def room_add_note(room_id: str, request: Request):
    """Write a personal thought into a room."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid JSON: {exc}"})
    text = (data.get("text") or "").strip() if isinstance(data, dict) else ""
    if not text:
        return JSONResponse(status_code=400, content={"error": "empty note"})
    return {"note": neo4j_store.add_note(room_id, text)}


@app.patch("/rooms/{room_id}/notes/{note_id}")
async def room_update_note(room_id: str, note_id: str, request: Request):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid JSON: {exc}"})
    text = (data.get("text") or "").strip() if isinstance(data, dict) else ""
    if not text:
        return JSONResponse(status_code=400, content={"error": "empty note"})
    note = neo4j_store.update_note(room_id, note_id, text)
    if note is None:
        return JSONResponse(status_code=404, content={"error": "note not found"})
    return {"note": note}


@app.delete("/rooms/{room_id}/notes/{note_id}")
async def room_delete_note(room_id: str, note_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    if not neo4j_store.delete_note(room_id, note_id):
        return JSONResponse(status_code=404, content={"error": "note not found"})
    return {"deleted": True}


@app.put("/events/{event_id}/room")
async def event_set_room(event_id: str, request: Request):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        data = await request.json()
        room_id = str(data.get("room_id") or "").strip()
        mode = str(data.get("mode") or "primary")
        if not room_id:
            raise ValueError("room_id is required")
        if not neo4j_store.set_event_room(event_id, room_id, mode=mode):
            return JSONResponse(status_code=404, content={"error": "event not found"})
        return {"event_id": event_id, "room_id": room_id, "mode": mode}
    except (AttributeError, TypeError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.delete("/events/{event_id}/rooms/{room_id}")
async def event_remove_room(event_id: str, room_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        if not neo4j_store.remove_event_room(event_id, room_id):
            return JSONResponse(status_code=404, content={"error": "assignment not found"})
        return {"deleted": True}
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


def _room_chat_turn(room_id, message):
    """Build the room-chat prompt + citations. Shared by both room chat endpoints."""
    room = neo4j_store.get_room(room_id) or {"name": room_id}
    neo4j_store.add_message(room_id, "user", message)
    ctx = neo4j_store.room_context(room_id)
    history = neo4j_store.room_messages(room_id, limit=10)

    # Retrieve on the actual question, scoped to this room, instead of pasting
    # whatever happened to be recent. Falls back to the room's newest events when
    # the question carries no content terms ("what have I been up to in here?").
    relevant = evidence_retriever.retrieve(
        message, limit=8, kinds=["event", "note", "claim"], room_id=room_id)
    citations = [{
        "number": index + 1, "kind": item.get("kind"), "id": item.get("id"),
        "title": item.get("title"), "text": item.get("text"), "ts": item.get("ts"),
    } for index, item in enumerate(relevant) if item.get("text")]
    relevant_lines = [
        f"[{item['number']}] ({item['kind']}) {item['text']}" for item in citations]

    grounding = (
        f"You are the user's assistant, chatting inside the '{room.get('name', room_id)}' room. "
        "This room collects the user's activity, notes, and your past chat on this topic. "
        "Use the context to answer; be concise and specific. " + INITIATIVE_PROMPT + "\n\n"
        f"Most relevant to this question:\n- " + "\n- ".join(relevant_lines or ["(none)"]) + "\n\n"
        f"Recent activity in this room:\n- " + "\n- ".join(ctx["events"][:8] or ["(none)"]) + "\n\n"
        f"User's notes here:\n- " + "\n- ".join(ctx["notes"][:8] or ["(none)"]) + "\n\n"
        f"Key things seen here: {', '.join(ctx['entities'][:15]) or '(none)'}"
    )
    if room.get("instructions"):
        grounding += f"\n\nRoom-specific user instructions:\n{room['instructions']}"
    messages = [{"role": "system", "content": grounding}]
    for m in history[:-1]:  # prior turns (exclude the just-added user message)
        messages.append({"role": m["role"], "content": m["text"]})
    messages.append({"role": "user", "content": message})
    return messages, citations


@app.post("/rooms/{room_id}/chat")
async def room_chat(room_id: str, request: Request):
    """Chat with the assistant scoped to a room (grounded in its events/notes)."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    message, error = await _read_message(request)
    if error is not None:
        return error

    messages, citations = _room_chat_turn(room_id, message)
    try:
        resp = await client.chat.completions.create(
            model=vlm_model, messages=messages, max_tokens=500)
        reply = resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("room_chat LLM failed: %s", exc)
        return JSONResponse(status_code=502, content={"error": f"chat failed: {exc}"})

    neo4j_store.add_message(room_id, "assistant", reply)
    return {"room_id": room_id, "reply": reply, "citations": citations}


@app.post("/rooms/{room_id}/chat/stream")
async def room_chat_stream(room_id: str, request: Request):
    """Room chat, streamed: citations first, then tokens."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    message, error = await _read_message(request)
    if error is not None:
        return error

    messages, citations = _room_chat_turn(room_id, message)

    def persist(reply):
        return neo4j_store.add_message(room_id, "assistant", reply)

    return StreamingResponse(
        _stream_reply(messages, citations, persist, max_tokens=500),
        media_type="application/x-ndjson")


@app.get("/memory/timeline")
async def memory_timeline(date: str = None, domain: str = None):
    """Sessions -> events with spans for a day. Neo4j-backed, JSONL fallback."""
    if domain not in {None, "personal", "home"}:
        return JSONResponse(
            status_code=400, content={"error": "domain must be personal or home"})
    ds = date or _today_iso()
    if neo4j_store is not None:
        try:
            rows = neo4j_store.sessions_with_events(ds, domain=domain)
            sessions = []
            for s in rows:
                events = [e for e in (s.get("events") or []) if e]
                events.sort(key=lambda e: e.get("span_start") or 0)
                sessions.append({
                    "session_id": s.get("session_id"),
                    "application": s.get("application"),
                    "activity_type": s.get("activity"),
                    "project_id": s.get("project_id"),
                    "state": s.get("state"),
                    "active_seconds": s.get("active_seconds"),
                    "resume_count": s.get("resume_count"),
                    "events": events,
                })
            return {"date": ds, "source": "neo4j", "sessions": sessions}
        except Exception as exc:
            logger.warning("memory_timeline (neo4j) failed: %s", exc)
    # Fallback: today's JSONL (only meaningful for today).
    return await debug_timeline()


@app.get("/memory/entities")
async def memory_entities(date: str = None, limit: int = 40,
                          domain: str = None):
    """Top entities for a day."""
    if domain not in {None, "personal", "home"}:
        return JSONResponse(
            status_code=400, content={"error": "domain must be personal or home"})
    if neo4j_store is None:
        return {"error": "graph not enabled"}
    ds = date or _today_iso()
    return {"date": ds, "domain": domain,
            "entities": neo4j_store.day_entities(
                ds, limit=limit, domain=domain)}


@app.get("/memory/entity")
async def memory_entity(name: str):
    """An entity's event history + same-frame co-occurrences."""
    if neo4j_store is None:
        return {"error": "graph not enabled"}
    return {"entity": name,
            "events": neo4j_store.events_for_entity(name),
            "co_occurring": neo4j_store.co_occurring_entities(name)}


@app.get("/memory/search")
async def memory_search(q: str, limit: int = 40, kinds: str = None,
                        from_date: str = None, to_date: str = None,
                        room_id: str = None, domain: str = None,
                        semantic: bool = True):
    """Unified graph + semantic search across all user-visible memory types."""
    if domain not in {None, "personal", "home"}:
        return JSONResponse(
            status_code=400, content={"error": "domain must be personal or home"})
    if not q.strip():
        return {"query": q, "results": []}
    selected = [v.strip() for v in kinds.split(",") if v.strip()] if kinds else None
    start = end = None
    try:
        if from_date:
            start = datetime.datetime.fromisoformat(from_date).timestamp()
        if to_date:
            end = (datetime.datetime.fromisoformat(to_date)
                   + datetime.timedelta(days=1)).timestamp()
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid date: {exc}"})

    # Semantic hits are scoped by the same room/date window as the keyword ones
    # (see EvidenceRetriever), so filters no longer disable vector search.
    results = evidence_retriever.retrieve(
        q, limit=limit, kinds=selected, start=start, end=end,
        room_id=room_id, domain=domain, semantic=semantic)
    return {"query": q, "domain": domain, "results": results}


@app.get("/memory/events/{event_id}")
async def memory_event_detail(event_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    event = neo4j_store.event_detail(event_id)
    if event is None:
        return JSONResponse(status_code=404, content={"error": "event not found"})
    return {"event": event}


@app.patch("/memory/events/{event_id}")
async def memory_event_update(event_id: str, request: Request):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    data = await request.json()
    if not isinstance(data, dict):
        return JSONResponse(status_code=400, content={"error": "body must be an object"})
    if "flagged" in data and not isinstance(data["flagged"], bool):
        return JSONResponse(status_code=400, content={"error": "flagged must be true or false"})
    summary = ((data.get("summary") or "").strip()
               if "summary" in data else None)
    has_triage = any(key in data for key in ("priority", "flagged", "flag_reason"))
    if summary == "":
        return JSONResponse(status_code=400, content={"error": "summary cannot be empty"})
    if summary is None and not has_triage:
        return JSONResponse(status_code=400, content={"error": "no event changes supplied"})
    try:
        event = (neo4j_store.update_event_summary(event_id, summary)
                 if summary is not None else neo4j_store.event_detail(event_id))
        if event is not None and has_triage:
            event = neo4j_store.update_event_metadata(
                event_id,
                priority=data.get("priority") if "priority" in data else None,
                flagged=data.get("flagged") if "flagged" in data else None,
                flag_reason=data.get("flag_reason") if "flag_reason" in data else None)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if event is None:
        return JSONResponse(status_code=404, content={"error": "event not found"})
    detail = neo4j_store.event_detail(event_id)
    if summary is not None and activity_logger is not None and detail:
        try:
            activity_logger.log_event(
                summary=summary, event_id=event_id,
                session_id=detail.get("session_id"),
                span_start=detail.get("span_start"), span_end=detail.get("span_end"),
                timestamp=detail.get("span_start"))
        except Exception as exc:
            logger.warning("event re-embedding failed: %s", exc)
    return {"event": event}


@app.delete("/memory/events/{event_id}")
async def memory_event_forget(event_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    result = neo4j_store.forget_event(event_id)
    if not result or not result.get("deleted"):
        return JSONResponse(status_code=404, content={"error": "event not found"})
    if activity_logger is not None:
        try:
            activity_logger.delete_event(event_id)
        except Exception as exc:
            logger.warning("event vector deletion failed: %s", exc)
    return result


@app.get("/memory/entities/{entity_id}")
async def memory_entity_detail(entity_id: str, domain: str = None):
    if domain not in {None, "personal", "home"}:
        return JSONResponse(
            status_code=400, content={"error": "domain must be personal or home"})
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    entity = neo4j_store.entity_detail(entity_id, domain=domain)
    if entity is None:
        return JSONResponse(status_code=404, content={"error": "entity not found"})
    return {"entity": entity}


@app.patch("/memory/entities/{entity_id}")
async def memory_entity_update(entity_id: str, request: Request):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    data = await request.json()
    name = str(data.get("name")).strip() if data.get("name") is not None else None
    entity_type = (str(data.get("type")).strip()
                   if data.get("type") is not None else None)
    if name == "" or entity_type == "":
        return JSONResponse(status_code=400, content={"error": "values cannot be empty"})
    entity = neo4j_store.update_entity(entity_id, name=name, entity_type=entity_type)
    if entity is None:
        return JSONResponse(status_code=404, content={"error": "entity not found"})
    return {"entity": entity}


@app.delete("/memory/entities/{entity_id}")
async def memory_entity_forget(entity_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    if not neo4j_store.forget_entity(entity_id):
        return JSONResponse(status_code=404, content={"error": "entity not found"})
    return {"deleted": True, "entity_id": entity_id}


@app.post("/memory/entities/{entity_id}/merge")
async def memory_entity_merge(entity_id: str, request: Request):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    data = await request.json()
    target_id = str(data.get("target_id") or "").strip()
    if not target_id:
        return JSONResponse(status_code=400, content={"error": "target_id is required"})
    try:
        result = neo4j_store.merge_entities(entity_id, target_id)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if result is None:
        return JSONResponse(status_code=404, content={"error": "source or target not found"})
    return result


@app.post("/memory/entities/{entity_id}/split")
async def memory_entity_split(entity_id: str, request: Request):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    data = await request.json()
    name = str(data.get("name") or "").strip()
    entity_type = str(data.get("type") or "").strip()
    event_ids = data.get("event_ids") if isinstance(data.get("event_ids"), list) else []
    if not name or not entity_type or not event_ids:
        return JSONResponse(
            status_code=400,
            content={"error": "name, type and event_ids are required"})
    from memory.rooms.registry import _slug
    new_entity_id = str(data.get("entity_id") or _slug(name))
    try:
        result = neo4j_store.split_entity(
            entity_id, new_entity_id, name, entity_type,
            [str(value) for value in event_ids])
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if result is None:
        return JSONResponse(status_code=404, content={"error": "source entity not found"})
    return result


@app.patch("/memory/claims/{claim_id}")
async def memory_claim_update(claim_id: str, request: Request):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    data = await request.json()
    text = (data.get("text") or "").strip() if isinstance(data, dict) else ""
    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})
    claim = neo4j_store.update_claim(claim_id, text)
    if claim is None:
        return JSONResponse(status_code=404, content={"error": "claim not found"})
    return {"claim": claim}


@app.delete("/memory/claims/{claim_id}")
async def memory_claim_forget(claim_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    if not neo4j_store.delete_claim(claim_id):
        return JSONResponse(status_code=404, content={"error": "claim not found"})
    return {"deleted": True, "claim_id": claim_id}


@app.delete("/memory/sessions/{session_id}")
async def memory_session_forget(session_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    result = neo4j_store.forget_session(session_id)
    for event_id in result["event_ids"]:
        if activity_logger is not None:
            try:
                activity_logger.delete_event(event_id)
            except Exception as exc:
                logger.warning("session vector deletion failed: %s", exc)
    return result


@app.delete("/memory/days/{date}")
async def memory_day_forget(date: str, domain: str = None):
    if domain not in {None, "personal", "home"}:
        return JSONResponse(
            status_code=400, content={"error": "domain must be personal or home"})
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        datetime.date.fromisoformat(date)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid date: {exc}"})
    result = neo4j_store.forget_day(date, domain=domain)
    for event_id in result["event_ids"]:
        if activity_logger is not None:
            try:
                activity_logger.delete_event(event_id)
            except Exception as exc:
                logger.warning("day vector deletion failed: %s", exc)
    return result


# === PHASE 3: GROUNDED ASSISTANT, REVIEWS, AND FOCUS ======================
@app.get("/assistant/conversations")
async def assistant_conversations():
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    return {"conversations": neo4j_store.list_conversations()}


@app.post("/assistant/conversations")
async def assistant_conversation_create(request: Request):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    data = await request.json()
    scope = str(data.get("scope") or "all")
    if scope not in {"all", "room", "today", "range"}:
        return JSONResponse(status_code=400, content={"error": "invalid scope"})
    room_id = data.get("room_id")
    if scope == "room" and not room_id:
        return JSONResponse(status_code=400, content={"error": "room_id is required"})
    conversation = neo4j_store.create_conversation(
        title=str(data.get("title") or "New conversation"),
        scope=scope, room_id=room_id,
        from_ts=data.get("from_ts"), to_ts=data.get("to_ts"))
    return JSONResponse(status_code=201, content={"conversation": conversation})


@app.get("/assistant/conversations/{conversation_id}")
async def assistant_conversation_get(conversation_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    conversation = neo4j_store.get_conversation(conversation_id)
    if conversation is None:
        return JSONResponse(status_code=404, content={"error": "conversation not found"})
    return {"conversation": conversation}


@app.patch("/assistant/conversations/{conversation_id}")
async def assistant_conversation_update(conversation_id: str, request: Request):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    data = await request.json()
    conversation = neo4j_store.update_conversation(conversation_id, data)
    if conversation is None:
        return JSONResponse(status_code=404, content={"error": "conversation not found"})
    return {"conversation": conversation}


@app.delete("/assistant/conversations/{conversation_id}")
async def assistant_conversation_delete(conversation_id: str):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    if not neo4j_store.delete_conversation(conversation_id):
        return JSONResponse(status_code=404, content={"error": "conversation not found"})
    return {"deleted": True}


def _assistant_scope(conversation):
    """(start, end, room_id) for a conversation's declared memory scope."""
    start, end, room_id = conversation.get("from_ts"), conversation.get("to_ts"), None
    if conversation.get("scope") == "today":
        start = datetime.datetime.combine(
            datetime.date.today(), datetime.time.min).timestamp()
        end = start + 86400
    elif conversation.get("scope") == "room":
        room_id = conversation.get("room_id")
    return start, end, room_id


def _assistant_turn(conversation, message):
    """Retrieve evidence and build the prompt. Shared by the blocking and
    streaming endpoints so both answer identically."""
    start, end, room_id = _assistant_scope(conversation)
    evidence = evidence_retriever.retrieve(
        message, limit=10, kinds=["event", "note", "claim"],
        start=start, end=end, room_id=room_id)
    citations = [{
        "number": index + 1, "kind": item.get("kind"), "id": item.get("id"),
        "title": item.get("title"), "text": item.get("text"),
        "ts": item.get("ts"), "rooms": item.get("rooms") or [],
    } for index, item in enumerate(evidence)]
    evidence_text = "\n".join(
        f"[{item['number']}] ({item['kind']}) {item['text']}"
        for item in citations) or "(No matching stored memory was found.)"
    system = (
        "You are the user's private, local-first assistant. Answer only from the "
        "provided memory evidence and ordinary reasoning. Never invent remembered "
        "facts. Cite supporting memories inline using [1], [2], etc. If evidence "
        "is insufficient, say so clearly. " + INITIATIVE_PROMPT
        + "\n\nMemory evidence:\n" + evidence_text
    )
    if room_id:
        room = neo4j_store.get_room(room_id)
        if room and room.get("instructions"):
            system += "\n\nRoom-specific user instructions:\n" + room["instructions"]
    messages = [{"role": "system", "content": system}]
    for prior in conversation.get("messages", [])[-12:]:
        if prior.get("role") in {"user", "assistant"}:
            messages.append({"role": prior["role"], "content": prior["text"]})
    messages.append({"role": "user", "content": message})
    return messages, citations


async def _read_message(request):
    """(message, error_response) from a JSON body with a `message` field."""
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, JSONResponse(status_code=400,
                                  content={"error": f"invalid JSON: {exc}"})
    message = (data.get("message") or "").strip() if isinstance(data, dict) else ""
    if not message:
        return None, JSONResponse(status_code=400,
                                  content={"error": "message is required"})
    return message, None


@app.post("/assistant/conversations/{conversation_id}/messages")
async def assistant_conversation_message(conversation_id: str, request: Request):
    """Answer from an explicit memory scope and return inspectable citations."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    message, error = await _read_message(request)
    if error is not None:
        return error
    conversation = neo4j_store.get_conversation(conversation_id, message_limit=30)
    if conversation is None:
        return JSONResponse(status_code=404, content={"error": "conversation not found"})

    messages, citations = _assistant_turn(conversation, message)
    neo4j_store.add_conversation_message(conversation_id, "user", message)
    try:
        response = await client.chat.completions.create(
            model=vlm_model, messages=messages, max_tokens=700)
        reply = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("grounded assistant failed: %s", exc)
        return JSONResponse(status_code=502, content={"error": f"assistant failed: {exc}"})
    saved = neo4j_store.add_conversation_message(
        conversation_id, "assistant", reply, citations=citations)
    return {"reply": reply, "citations": citations, "message": saved}


async def _stream_reply(messages, citations, on_complete, max_tokens=700):
    """NDJSON token stream, citations first.

    Evidence is known before generation starts, so sending it immediately lets
    the UI render sources while the answer is still being written — the whole
    perceived-latency win. Matches the existing /talk NDJSON convention.
    """
    yield json.dumps({"type": "citations", "citations": citations}) + "\n"
    parts = []
    try:
        stream = await client.chat.completions.create(
            model=vlm_model, messages=messages, max_tokens=max_tokens, stream=True)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                parts.append(piece)
                yield json.dumps({"type": "delta", "text": piece}) + "\n"
    except Exception as exc:
        logger.warning("streaming assistant failed: %s", exc)
        yield json.dumps({"type": "error", "error": str(exc)}) + "\n"
        return

    reply = "".join(parts).strip()
    saved = None
    try:
        saved = on_complete(reply)
    except Exception as exc:
        logger.warning("persisting streamed reply failed: %s", exc)
    yield json.dumps({"type": "done", "reply": reply, "message": saved}) + "\n"


@app.post("/assistant/conversations/{conversation_id}/messages/stream")
async def assistant_conversation_message_stream(conversation_id: str, request: Request):
    """Same answer as the blocking endpoint, streamed: citations, then tokens."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    message, error = await _read_message(request)
    if error is not None:
        return error
    conversation = neo4j_store.get_conversation(conversation_id, message_limit=30)
    if conversation is None:
        return JSONResponse(status_code=404, content={"error": "conversation not found"})

    messages, citations = _assistant_turn(conversation, message)
    neo4j_store.add_conversation_message(conversation_id, "user", message)

    def persist(reply):
        return neo4j_store.add_conversation_message(
            conversation_id, "assistant", reply, citations=citations)

    return StreamingResponse(
        _stream_reply(messages, citations, persist),
        media_type="application/x-ndjson")


@app.get("/memory/aliases")
async def memory_aliases():
    """Naming corrections learned from merges/renames, applied to future captures."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    return {"aliases": neo4j_store.list_entity_aliases(),
            "naming_hints": [{"wrong": wrong, "canonical": canonical}
                             for wrong, canonical in neo4j_store.canonical_name_hints()]}


@app.delete("/memory/aliases/{alias_id:path}")
async def memory_alias_delete(alias_id: str):
    """Undo a learned alias (the entity will be treated as distinct again)."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    if not neo4j_store.delete_entity_alias(alias_id):
        return JSONResponse(status_code=404, content={"error": "alias not found"})
    return {"deleted": True, "alias_id": alias_id}


@app.get("/reviews/daily")
async def review_daily(date: str = None):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    from memory.summary.coach import format_report
    day = date or _today_iso()
    metrics = neo4j_store.daily_metrics(day)
    claims = neo4j_store.day_claims(day, limit=8)
    entities = neo4j_store.day_entities(day, limit=12)
    return {"date": day, "metrics": metrics, "claims": claims,
            "entities": entities,
            "report": format_report(metrics, claims=claims, entities=entities)}


@app.get("/reviews/weekly")
async def review_weekly(end_date: str = None):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        end = datetime.date.fromisoformat(end_date) if end_date else datetime.date.today()
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid date: {exc}"})
    days = []
    for offset in range(6, -1, -1):
        day = end - datetime.timedelta(days=offset)
        days.append(neo4j_store.daily_metrics(day.isoformat()))
    active = sum(day.get("active_minutes") or 0 for day in days)
    events = sum(day.get("events") or 0 for day in days)
    switches = sum(day.get("switches") or 0 for day in days)
    focus_days = [day.get("focus_score") or 0 for day in days if day.get("events")]
    return {
        "start_date": days[0]["date"], "end_date": days[-1]["date"], "days": days,
        "summary": {
            "active_minutes": round(active, 1), "events": events,
            "switches": switches,
            "average_focus_score": round(sum(focus_days) / len(focus_days))
            if focus_days else 0,
        },
    }


@app.get("/focus/sessions")
async def focus_sessions():
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    return {"active": neo4j_store.active_focus_session(),
            "sessions": neo4j_store.list_focus_sessions()}


@app.post("/focus/sessions")
async def focus_session_start(request: Request):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    if neo4j_store.active_focus_session() is not None:
        return JSONResponse(status_code=409, content={"error": "a focus session is already active"})
    data = await request.json()
    goal = (data.get("goal") or "").strip() if isinstance(data, dict) else ""
    if not goal:
        return JSONResponse(status_code=400, content={"error": "goal is required"})
    try:
        minutes = max(5, min(int(data.get("planned_minutes") or 25), 240))
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": "planned_minutes must be an integer"})
    focus = neo4j_store.start_focus_session(
        goal, room_id=data.get("room_id"), planned_minutes=minutes)
    return JSONResponse(status_code=201, content={"focus": focus})


@app.post("/focus/sessions/{focus_id}/stop")
async def focus_session_stop(focus_id: str, recap: bool = True):
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    focus = neo4j_store.stop_focus_session(focus_id)
    if focus is None:
        return JSONResponse(status_code=404, content={"error": "active focus session not found"})
    if recap:
        focus["recap"] = await build_focus_recap(focus, post=True)
    return {"focus": focus}


@app.get("/focus/sessions/{focus_id}/recap")
async def focus_session_recap(focus_id: str, regenerate: bool = False,
                              post: bool = False):
    """The stored recap, or a freshly built one (works for past sessions too)."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    focus = neo4j_store.get_focus_session(focus_id)
    if focus is None:
        return JSONResponse(status_code=404, content={"error": "focus session not found"})
    if not regenerate and focus.get("recap"):
        return {"focus_id": focus_id, "recap": focus["recap"], "cached": True}
    return {"focus_id": focus_id,
            "recap": await build_focus_recap(focus, post=post), "cached": False}


async def build_focus_recap(focus, post=False):
    """Classify a finished session's events against its goal, then render a recap.

    Attribution is by time overlap (see _FOCUS_EVENTS_CYPHER), so this also works
    for sessions that ran before recaps existed.
    """
    from memory.summary import focus_recap as recap_mod

    start = focus.get("started_at")
    end = focus.get("ended_at") or time.time()
    goal = focus.get("goal") or "(no goal recorded)"
    if start is None:
        return None

    events = neo4j_store.focus_events(start, end, room_id=focus.get("room_id"))
    if not events:
        breakdown = recap_mod.apply_classification(goal, [], {}, start, end)
        return recap_mod.format_recap(breakdown, focus.get("planned_minutes"))

    labels = {}
    try:
        response = await client.chat.completions.create(
            model=vlm_model, max_tokens=600,
            messages=[{"role": "user",
                       "content": recap_mod.classify_prompt(goal, events)}])
        labels = recap_mod.parse_labels(
            response.choices[0].message.content, len(events))
    except Exception as exc:
        # Unlabelled events fall into "unknown", which the recap reports honestly
        # rather than scoring the session as a failure.
        logger.warning("focus classification failed: %s", exc)

    breakdown = recap_mod.apply_classification(goal, events, labels, start, end)

    feedback = ""
    if breakdown["on_task_pct"] is not None:
        try:
            response = await client.chat.completions.create(
                model=vlm_model, max_tokens=250,
                messages=[{"role": "user",
                           "content": recap_mod.feedback_prompt(breakdown)}])
            feedback = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("focus feedback LLM failed: %s", exc)

    text = recap_mod.format_recap(
        breakdown, focus.get("planned_minutes"), feedback=feedback)
    try:
        neo4j_store.save_focus_recap(focus["focus_id"], text, breakdown)
        if post:
            neo4j_store.add_message(
                focus.get("room_id") or "daily", "coach", text, ts=end)
    except Exception as exc:
        logger.warning("saving focus recap failed: %s", exc)
    return text


@app.get("/debug/co-occurrence")
async def debug_co_occurrence(entity: str):
    """Step 11: entities that appeared in the SAME FRAME as `entity`."""
    if neo4j_store is None:
        return {"error": "graph not enabled (start with MEMORY_NEO4J=1)"}
    return {"entity": entity,
            "co_occurring": neo4j_store.co_occurring_entities(entity)}


@app.get("/debug/entity")
async def debug_entity(name: str):
    """Step 11: an entity's event timeline + same-frame co-occurrences."""
    if neo4j_store is None:
        return {"error": "graph not enabled (start with MEMORY_NEO4J=1)"}
    return {"entity": name,
            "events": neo4j_store.events_for_entity(name),
            "co_occurring": neo4j_store.co_occurring_entities(name)}


@app.get("/debug/hybrid")
async def debug_hybrid(q: str, limit: int = 5):
    """Step 11: hybrid retrieval — vector search over event summaries (Qdrant),
    enriched with each event's graph entities (Neo4j)."""
    from qdrant_client import models as qmodels
    flt = qmodels.Filter(must_not=[
        qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="session_id"))])
    try:
        hits = qdrant_client.query("activity_log", query_text=q,
                                   query_filter=flt, limit=limit)
    except Exception as exc:
        return {"error": f"vector search failed: {exc}"}
    results, ids = [], []
    for h in hits:
        m = h.metadata or {}
        ids.append(m.get("event_id"))
        results.append({
            "event_id": m.get("event_id"), "session_id": m.get("session_id"),
            "span_start": m.get("span_start"), "span_end": m.get("span_end"),
            "profile": m.get("profile"), "summary": (m.get("document") or "")[:200],
        })
    enrich = neo4j_store.entities_for_events(ids) if neo4j_store is not None else {}
    for r in results:
        r["entities"] = enrich.get(r["event_id"], [])
    return {"query": q, "results": results}


@app.post("/debug/consolidate")
async def debug_consolidate(min_events: int = 2):
    """Step 14: promote/quarantine entities + rebuild shortcut edges."""
    if neo4j_store is None:
        return {"error": "graph not enabled (start with MEMORY_NEO4J=1)"}
    status = neo4j_store.consolidate(min_events=min_events)
    shortcuts = neo4j_store.rebuild_shortcuts()
    return {"status": status, "shortcuts": shortcuts,
            "status_counts": neo4j_store.status_counts()}


@app.post("/debug/resolve")
async def debug_resolve():
    """Step 12: run entity resolution now and return the alias candidates."""
    if neo4j_store is None:
        return {"error": "graph not enabled (start with MEMORY_NEO4J=1)"}
    n = neo4j_store.resolve_entities()
    return {"candidates": n, "possibly_same_as": neo4j_store.possibly_same_as()}


@app.get("/debug/possibly-same-as")
async def debug_possibly_same_as():
    """Step 12: list current POSSIBLY_SAME_AS alias candidates."""
    if neo4j_store is None:
        return {"error": "graph not enabled (start with MEMORY_NEO4J=1)"}
    return {"possibly_same_as": neo4j_store.possibly_same_as()}


@app.post("/debug/daily-note")
async def debug_daily_note(date: str = None, resolve: bool = True):
    """Step 13: generate today's (or `date`) Obsidian note + Pending-Merges."""
    if neo4j_store is None:
        return {"error": "graph not enabled (start with MEMORY_NEO4J=1)"}
    from memory.summary.daily_summarizer import export_to_vault
    if resolve:
        neo4j_store.resolve_entities()
    vault = os.getenv("OBSIDIAN_VAULT", "obsidian_notes")
    paths = export_to_vault(neo4j_store, vault, date)
    return {"written": paths}


@app.post("/history/clear")
async def clear_history_endpoint():
    """One-shot clear of the in-memory conversation."""
    _chat_history.clear()
    logger.info("Conversation history cleared.")
    return {"cleared": True, "history_turns": 0}


@app.post("/memory/clear")
async def clear_memory_endpoint():
    """POC debug control: drop the long-term activity collection."""
    try:
        activity_logger.reset()
        logger.info("Activity memory collection cleared.")
        return {"cleared": True}
    except Exception as exc:
        logger.warning("Failed to clear memory: %s", exc)
        return {"cleared": False, "error": str(exc)}


@app.get("/proactive")
async def proactive_insights(since: int = 0):
    """Proactive insights newer than `since` (by id), each with base64 TTS audio
    so the end device can play them. The client tracks the last id it has seen
    and passes it as `since` to receive only new insights."""
    items = [i for i in _proactive_insights if i["id"] > since]
    return {"enabled": proactive is not None, "latest_id": _proactive_seq, "insights": items}


@app.get("/notifications")
async def notifications_list(since: int = 0, limit: int = 100,
                             unread_only: bool = False):
    """Durable critical/important event inbox, newest first."""
    return notification_center.list(
        since=since, limit=limit, unread_only=unread_only)


@app.post("/notifications/{notification_id}/read")
async def notification_mark_read(notification_id: str):
    item = notification_center.mark_read(notification_id)
    if item is None:
        return JSONResponse(status_code=404, content={"error": "notification not found"})
    return {"notification": item}


@app.post("/notifications/actions/read-all")
async def notifications_mark_all_read():
    return {"updated": notification_center.mark_all_read()}


NUDGE_FEEDBACK_VALUES = {"up", "down", "not_now"}


@app.post("/proactive/{nudge_id}/feedback")
async def proactive_feedback(nudge_id: str, request: Request):
    """Record how the user reacted, so the narrator stops repeating rejected themes.

    Without this the narrator can only get louder, never more selective.
    """
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid JSON: {exc}"})
    feedback = (data.get("feedback") or "").strip() if isinstance(data, dict) else ""
    if feedback not in NUDGE_FEEDBACK_VALUES:
        return JSONResponse(status_code=400, content={
            "error": f"feedback must be one of {sorted(NUDGE_FEEDBACK_VALUES)}"})
    updated = neo4j_store.set_nudge_feedback(nudge_id, feedback)
    if updated is None:
        return JSONResponse(status_code=404, content={"error": "nudge not found"})
    return {"nudge": updated}


@app.get("/proactive/history")
async def proactive_history(limit: int = 50):
    """Past nudges with their feedback (for tuning what the narrator says)."""
    if neo4j_store is None:
        return JSONResponse(status_code=400, content={"error": "graph not enabled"})
    return {"nudges": neo4j_store.list_nudges(limit=limit)}


def handle_observation_description(description, timestamp, source="screen", context=None):
    """Capture-thread callback: ask the narrator for an insight, synthesize its
    speech, and queue it for the end device to play (no server-side playback).
    May be called by desktop, mobile, or camera workers."""
    global _proactive_seq
    if proactive is None:
        return
    try:
        insight = asyncio.run(
            proactive.consider(description, source=source, context=context))
    except Exception as exc:
        logger.warning("Proactive consider failed: %s", exc)
        return
    if not insight:
        return
    text = insight["text"]
    # Synthesize speech here so the client can play it on the end device.
    audio_b64 = None
    try:
        audio_b64 = base64.b64encode(run_kokoro(text)).decode("utf-8")
    except Exception as exc:
        logger.warning("Proactive TTS failed: %s", exc)
    # Persist it so the user's reaction can be fed back into future nudges.
    nudge_id = None
    if neo4j_store is not None:
        try:
            nudge_id = neo4j_store.record_nudge(
                text, kind=insight.get("kind"), focus_id=insight.get("focus_id"),
                evidence=insight.get("evidence"))
        except Exception as exc:
            logger.warning("Recording nudge failed: %s", exc)
    _proactive_seq += 1
    _proactive_insights.append({
        "id": _proactive_seq,
        "nudge_id": nudge_id,
        "text": text,
        "kind": insight.get("kind"),
        "source": insight.get("source") or source,
        "focus_id": insight.get("focus_id"),
        "evidence": insight.get("evidence") or [],
        "timestamp": timestamp,
        "audio": audio_b64,
    })
    logger.info("Proactive insight #%d (%s): %s",
                _proactive_seq, insight.get("kind"), text)


def handle_screen_description(description, timestamp):
    """Backward-compatible desktop capture callback."""
    return handle_observation_description(
        description, timestamp, source="desktop_screen")


async def describe_mobile_frames(source, frames):
    """Create one memory description from a mobile capture window."""
    content = [{
        "type": "text",
        "text": (
            f"Describe this sequence of {source} frames as a concise factual activity "
            "timeline. Include important visible text, actions, changes, and context "
            "that would be useful to remember later. Do not guess."
        ),
    }]
    # Bound prompt size independently from the live conversational frame limit.
    for frame in frames[-MAX_FRAMES:]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(frame)}"},
        })
    response = await client.chat.completions.create(
        model=vlm_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=800,
    )
    return (response.choices[0].message.content or "").strip()


async def process_mobile_activity():
    """Periodically connect mobile capture to memory and proactive narration."""
    interval = env_int("MOBILE_ACTIVITY_INTERVAL_SECONDS", 60, minimum=5)
    while True:
        await asyncio.sleep(interval)
        source, frames = mobile_stream.processing_window()
        if not frames or vlm_model is None:
            continue
        timestamp = time.time()
        try:
            description = await describe_mobile_frames(source, frames)
            if not description:
                continue
            await asyncio.to_thread(
                activity_logger.log_activity, description, timestamp, f"mobile_{source}", []
            )
            mobile_stream.processed()
            if proactive is not None:
                await asyncio.to_thread(
                    handle_observation_description, description, timestamp,
                    f"mobile_{source}", {"capture_source": source})
            logger.info("Processed %d mobile %s frames into memory", len(frames), source)
        except Exception as exc:
            mobile_stream.last_error = f"activity processing failed: {exc}"
            logger.exception("Mobile activity processing failed")


# === API ===
@app.post("/chat/audio")
async def live_chat(request: Request):
    global _current_context

    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid JSON: {exc}"})
    if not isinstance(data, dict):
        return JSONResponse(status_code=400, content={"error": "request body must be a JSON object"})

    wav_bytes_audio = data.get("data")
    wav_bytes_image = data.get("image")
    clear_history = data.get("clear_history", False)
    concise = data.get("talking", False)
    context = data.get("context") or "talker"
    live = data.get("live", False)
    memory = data.get("memory", False)
    # A typed turn: same pipeline, minus ASR. Everything downstream (live
    # frames, memory tools, TTS) behaves exactly as it does for speech.
    typed_text = (data.get("text") or "").strip()

    if not typed_text and not wav_bytes_audio:
        return JSONResponse(status_code=400,
                            content={"error": "provide either 'text' or audio 'data'"})

    _current_context = context

    if clear_history:
        _chat_history.clear()
        logger.info("Conversation history cleared (per-request flag).")

    return StreamingResponse(
        generate_response(
            wav_bytes_audio, wav_bytes_image, _chat_history,
            concise, context, live, memory, typed_text or None,
        ),
        media_type="application/x-ndjson",
    )


@app.post("/transcribe")
async def transcribe(request: Request):
    """Speech to text only, for surfaces that compose text rather than converse.

    The room composer records a clip and dictates into its input box, so the
    user can edit before deciding whether it becomes a note or a question.
    """
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return JSONResponse(status_code=400, content={"error": f"invalid JSON: {exc}"})
    if not isinstance(data, dict):
        return JSONResponse(status_code=400, content={"error": "request body must be a JSON object"})

    try:
        audio_data = decode_audio_to_array(data.get("data"))
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    try:
        t = time.perf_counter()
        text = await asyncio.to_thread(nemo_transcribe, audio_data)
        logger.info("dictation ASR %d ms: %s", int((time.perf_counter() - t) * 1000), text)
    except Exception as exc:
        logger.warning("dictation ASR failed: %s", exc)
        return JSONResponse(status_code=502, content={"error": f"transcription failed: {exc}"})
    return {"text": (text or "").strip()}


# === SMALL PIPELINE HELPERS ===
def ndjson(obj):
    """Serialize one NDJSON line."""
    return json.dumps(obj) + "\n"


def debug_line(turn_id, stage, **fields):
    """A debug/timing event, only emitted to the client when DEBUG_VERBOSE."""
    return ndjson({"type": "debug", "turn": turn_id, "stage": stage, **fields})


def error_line(turn_id, stage, message):
    return ndjson({"type": "error", "turn": turn_id, "stage": stage, "message": message})


def decode_audio_to_array(wav_bytes_audio):
    """Convert raw 16-bit PCM request bytes into a float array for the ASR model."""
    if not wav_bytes_audio:
        raise ValueError("no audio provided")

    raw = bytes(wav_bytes_audio)
    audio_seg = pydub.AudioSegment.from_raw(
        io.BytesIO(raw), sample_width=2, frame_rate=16000, channels=1
    )
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(audio_seg.channels)
        wav_file.setsampwidth(audio_seg.sample_width)
        wav_file.setframerate(audio_seg.frame_rate)
        wav_file.writeframesraw(audio_seg.raw_data)

    data, _ = sf.read(io.BytesIO(wav_io.getvalue()))
    if data.size == 0:
        raise ValueError("decoded audio is empty")
    return data


def _frames_for_context(context):
    """Return (frames_list, source_label, warning) for the requested live context."""
    if context == "screen":
        mobile_frames = mobile_stream.frames("screen")
        if mobile_frames:
            return mobile_frames, "mobile_screen", None
        if screen_stream is None or not screen_stream.status()["healthy"]:
            return [], "screen", "screen stream unavailable"
        return screen_stream.frames(), "screen", None
    if context == "camera":
        mobile_frames = mobile_stream.frames("camera")
        if mobile_frames:
            return mobile_frames, "mobile_camera", None
        # Use the first live camera's frames for the generic "camera" context.
        worker = None
        if camera_manager is not None:
            worker = next((w for w in camera_manager.workers.values()
                           if w.status().get("connected")), None)
        if worker is None:
            return [], "camera", "camera stream unavailable"
        return worker.stream.frames(), "camera", None
    return [], context, None


def build_user_content(transcription, image_b64, context, live):
    """Build the multimodal user message. Returns (content, info) where info
    reports the context source, frame count, and any warnings."""
    user_content = [{"type": "text", "text": transcription}]
    info = {"source": "text", "frames": 0, "warnings": []}

    # Single uploaded image (only when not pulling a live stream).
    if image_b64 and not live:
        try:
            Image.open(io.BytesIO(base64.b64decode(image_b64))).verify()
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            })
            info["source"] = "image"
            info["frames"] = 1
        except Exception as exc:
            info["warnings"].append(f"image decode failed: {exc}")

    # Live stream frames (screen/camera), capped at MAX_FRAMES.
    if live:
        frames, source, warning = _frames_for_context(context)
        if warning:
            info["warnings"].append(warning)
        if frames:
            frames = frames[-MAX_FRAMES:]
            for index, img in enumerate(frames):
                try:
                    encoded = encode_image_base64(img)
                except Exception as exc:
                    info["warnings"].append(
                        f"failed to encode {source} frame {index + 1}: {exc}"
                    )
                    continue
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "max_dynamic_patch": 9,
                        "url": f"data:image/jpeg;base64,{encoded}",
                    },
                })
        info["source"] = source
        info["frames"] = sum(
            1 for item in user_content if item.get("type") == "image_url"
        )

    return user_content, info


async def gather_tool_context(transcription):
    """Let the model pick tool(s), run them, and return their combined result
    as labeled context text. One round of tool calls (POC — no multi-hop loop).
    Returns (context_text or None, info)."""
    info = {"used": False, "tools": [], "items": 0}
    if not tool_registry:
        return None, info

    response = await client.chat.completions.create(
        model=vlm_model,
        messages=[
            {"role": "system",
             "content": "You have callable tools. Based on the user query, decide which tool(s) to call and with what arguments. If none are needed, do not call any."},
            {"role": "user", "content": transcription},
        ],
        tools=tool_registry.openai_schemas,
        tool_choice="auto",
    )

    tool_calls = response.choices[0].message.tool_calls or []
    if not tool_calls:
        logger.debug("Model requested no tools; answering directly.")
        return None, info

    blocks = []  # (tool_name, text)
    for call in tool_calls:
        tool = tool_registry.get(call.function.name)
        if tool is None:
            logger.warning("Model called unknown tool: %s", call.function.name)
            continue
        try:
            args = json.loads(call.function.arguments or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            logger.warning("Malformed args for %s: %s", tool.name, exc)
            continue
        try:
            result = await tool.run(**args)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", tool.name, exc)
            continue

        info["tools"].append(tool.name)
        if isinstance(result, list):
            blocks.extend((tool.name, str(text)) for text in result)
        elif result:
            blocks.append((tool.name, str(result)))

    blocks = blocks[:MAX_MEMORY_ITEMS]
    if not blocks:
        return None, info

    info["used"] = True
    info["items"] = len(blocks)
    labelled = "\n".join(f"[{name} {i + 1}] {text}" for i, (name, text) in enumerate(blocks))
    return labelled, info


def build_messages(concise, memory_text, chat_history, user_content):
    system_prompt = (
        CONCISE_SYSTEM_PROMPT if concise
        else "You are the user's personal assistant.\n\n" + INITIATIVE_PROMPT
    )
    messages = [{"role": "system", "content": system_prompt}]
    if memory_text:
        messages.append({
            "role": "user",
            "content": f"Here is some relevant past memory for context:\n{memory_text}",
        })
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_content})
    return messages


# === RESPONSE GENERATION ===
async def generate_response(wav_bytes_audio, wav_bytes_image, chat_history,
                            concise, context, live, memory, typed_text=None):
    """Handle an incoming turn (spoken or typed) and stream text + TTS as NDJSON."""
    turn_id = uuid.uuid4().hex[:8]
    t_turn = time.perf_counter()
    logger.info("[%s] turn start (context=%s live=%s memory=%s typed=%s)",
                turn_id, context, live, memory, bool(typed_text))
    set_pipeline_status(True, "typing" if typed_text else "transcribing", turn_id)

    if DEBUG_VERBOSE:
        yield debug_line(turn_id, "start", context=context, live=live, memory=memory)

    # 1. Get the user's words: either straight from the request, or via ASR.
    asr_ms = 0
    if typed_text:
        transcription = typed_text
    else:
        try:
            t = time.perf_counter()
            audio_data = decode_audio_to_array(wav_bytes_audio)
            transcription = nemo_transcribe(audio_data)
            asr_ms = int((time.perf_counter() - t) * 1000)
            logger.info("[%s] ASR %d ms: %s", turn_id, asr_ms, transcription)
        except Exception as exc:
            logger.warning("[%s] ASR failed: %s", turn_id, exc)
            set_pipeline_status(False, "asr_error", turn_id)
            yield error_line(turn_id, "asr", str(exc))
            return

    if vlm_model is None:
        set_pipeline_status(False, "model_unavailable", turn_id)
        yield error_line(turn_id, "model", "VLM model not loaded")
        return

    yield ndjson({"type": "query", "text": transcription})
    if DEBUG_VERBOSE:
        yield debug_line(turn_id, "asr", ms=asr_ms, text=transcription)

    # 2. Build multimodal user content.
    set_pipeline_status(True, "collecting_context", turn_id)
    user_content, ctx_info = build_user_content(transcription, wav_bytes_image, context, live)
    logger.info("[%s] context source=%s frames=%d warnings=%s",
                turn_id, ctx_info["source"], ctx_info["frames"], ctx_info["warnings"])
    for warning in ctx_info["warnings"]:
        yield error_line(turn_id, "context", warning)
    if DEBUG_VERBOSE:
        yield debug_line(turn_id, "context", source=ctx_info["source"], frames=ctx_info["frames"])

    # 3. Optional tool use (memory retrieval and any other registered tools).
    memory_text = None
    if memory:
        set_pipeline_status(True, "retrieving_memory", turn_id)
        try:
            t = time.perf_counter()
            memory_text, mem_info = await gather_tool_context(transcription)
            mem_ms = int((time.perf_counter() - t) * 1000)
            logger.info("[%s] tools %d ms: used=%s tools=%s items=%s",
                        turn_id, mem_ms, mem_info["used"], mem_info["tools"], mem_info["items"])
            if DEBUG_VERBOSE:
                yield debug_line(turn_id, "tools", ms=mem_ms, used=mem_info["used"],
                                 tools=mem_info["tools"], items=mem_info["items"])
        except Exception as exc:
            logger.warning("[%s] tool use failed: %s", turn_id, exc)
            yield error_line(turn_id, "tools", str(exc))

    # 4. Build the prompt and open the streaming VLM call.
    set_pipeline_status(True, "starting_model", turn_id)
    messages = build_messages(concise, memory_text, chat_history, user_content)
    try:
        chat_response = await client.chat.completions.create(
            model=vlm_model, messages=messages, stream=True,
        )
    except Exception as exc:
        logger.warning("[%s] VLM request failed: %s", turn_id, exc)
        set_pipeline_status(False, "model_error", turn_id)
        yield error_line(turn_id, "vlm", str(exc))
        return

    # 5. Stream text + audio.
    set_pipeline_status(True, "streaming_response", turn_id)
    t_first = None
    t_stream = time.perf_counter()
    full_assistant_response = ""
    async for line, text_chunk in stream_vlm_and_audio(chat_response, turn_id):
        if text_chunk and t_first is None:
            t_first = time.perf_counter()
            first_ms = int((t_first - t_turn) * 1000)
            logger.info("[%s] first token %d ms", turn_id, first_ms)
            if DEBUG_VERBOSE:
                yield debug_line(turn_id, "first_token", ms=first_ms)
        yield line
        if text_chunk:
            full_assistant_response += text_chunk

    vlm_stream_ms = int((time.perf_counter() - t_stream) * 1000)
    logger.info("[%s] response pipeline complete %d ms", turn_id, vlm_stream_ms)
    if DEBUG_VERBOSE:
        yield debug_line(turn_id, "stream_complete", ms=vlm_stream_ms)

    # 6. Persist the turn into the single shared history.
    chat_history.append({"role": "user", "content": user_content})
    chat_history.append({"role": "assistant", "content": full_assistant_response})

    total_ms = int((time.perf_counter() - t_turn) * 1000)
    logger.info("[%s] turn done %d ms", turn_id, total_ms)
    set_pipeline_status(False, "ready", turn_id)
    yield ndjson({"type": "done", "turn": turn_id, "total_ms": total_ms})


async def stream_vlm_and_audio(chat_response_stream, turn_id):
    """Stream VLM text sentence by sentence and batched TTS audio.
    Yields (ndjson_line_bytes, raw_text_for_history)."""
    full_sentence = ""
    sentence_buffer_for_audio = []
    audio_tasks = []  # (task, sentences_text)
    sentence_count = 0
    vlm_started = time.perf_counter()

    async def generate_audio_task(text):
        """Run TTS in a thread; never let one failed chunk break the response."""
        try:
            started = time.perf_counter()
            audio_bytes = await asyncio.to_thread(run_kokoro, text)
            tts_ms = int((time.perf_counter() - started) * 1000)
            logger.info("[%s] TTS %d ms (%d chars)", turn_id, tts_ms, len(text))
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            return {"type": "audio", "data": audio_base64, "tts_ms": tts_ms}
        except Exception as exc:
            logger.warning("[%s] TTS failed for %r: %s", turn_id, text, exc)
            return {"type": "error", "turn": turn_id, "stage": "tts", "message": str(exc)}

    def split_into_chunks(text, max_words=30):
        words = text.split()
        for i in range(0, len(words), max_words):
            yield " ".join(words[i:i + max_words])

    def emit_text(sentence_to_send):
        nonlocal sentence_count
        payload = {"type": "vlm_text", "text": sentence_to_send}
        sentence_buffer_for_audio.append(sentence_to_send)
        sentence_count += 1
        batch_size = 1 if sentence_count <= 3 else 3
        if len(sentence_buffer_for_audio) >= batch_size:
            text_to_speak = " ".join(sentence_buffer_for_audio)
            task = asyncio.create_task(generate_audio_task(text_to_speak))
            audio_tasks.append((task, text_to_speak))
            sentence_buffer_for_audio.clear()
        logger.debug("[%s] sent VLM text: %r", turn_id, sentence_to_send)
        return (ndjson(payload).encode("utf-8"), sentence_to_send)

    async for chunk in chat_response_stream:
        try:
            delta = chunk.choices[0].delta.content
        except (AttributeError, IndexError):
            delta = None
        if delta:
            full_sentence += delta
            if any(p in full_sentence for p in ".!?") or len(full_sentence.split()) >= 30:
                for sentence_to_send in split_into_chunks(full_sentence.strip()):
                    yield emit_text(sentence_to_send)
                full_sentence = ""

        while audio_tasks and audio_tasks[0][0].done():
            task, spoken_text = audio_tasks.pop(0)
            audio_payload = await task
            yield (ndjson(audio_payload).encode("utf-8"), None)

    vlm_ms = int((time.perf_counter() - vlm_started) * 1000)
    logger.info("[%s] VLM stream complete %d ms", turn_id, vlm_ms)
    if DEBUG_VERBOSE:
        yield (debug_line(turn_id, "vlm_complete", ms=vlm_ms).encode("utf-8"), None)

    # Flush any trailing text that never hit a sentence delimiter.
    if full_sentence.strip():
        for sentence_to_send in split_into_chunks(full_sentence.strip()):
            yield emit_text(sentence_to_send)

    # Flush remaining buffered sentences into a final audio task.
    if sentence_buffer_for_audio:
        text_to_speak = " ".join(sentence_buffer_for_audio)
        task = asyncio.create_task(generate_audio_task(text_to_speak))
        audio_tasks.append((task, text_to_speak))

    for task, spoken_text in audio_tasks:
        audio_payload = await task
        yield (ndjson(audio_payload).encode("utf-8"), None)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        log_level="info",
    )

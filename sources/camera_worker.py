"""Camera capture worker — one live RTSP camera → episodic memory.

Unlike the screen path (which keys sessions on app/window boundaries), a camera
has no "application" to switch between. So a worker treats the camera itself as
the session: every `window_seconds` it grabs the recent frames, asks the VLM for
a surveillance-focused structured extraction (people, vehicles, objects, and the
*events* between them — "the orange car was driven out of the garage"), and feeds
that through the shared MemoryPipeline. That writes an Event + entities/claims to
Neo4j and Qdrant exactly like the screen path, and routes the event into this
camera's own Room so its feed reads like a channel of what the camera saw.

Boundaries here come from visual change (a scene actually changing), never from
app switches — see MemoryPipeline. Pausing halts extraction without dropping the
RTSP connection, so it resumes instantly.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
import time
from threading import Event, Thread

import cv2
import numpy as np

from sources.rtsp import RealtimeCameraStream
from sources.motion_gate import MotionGate
from memory.models.extraction import ENTITY_TYPE_SUGGESTIONS
from memory.extraction.validator import run_extraction
from memory.pipeline import MemoryPipeline
from providers.local_openAI import new_vlm_client

logger = logging.getLogger("home_assistant")


# Entity vocabulary that matters for a physical scene, layered on the shared
# suggestions so the VLM still tags anything sensible it sees.
_CAMERA_ENTITY_TYPES = [
    "person", "vehicle", "car", "truck", "motorcycle", "bicycle",
    "animal", "package", "door", "bag", "tool", "object",
] + ENTITY_TYPE_SUGGESTIONS

_CAMERA_SYSTEM_PROMPT = f"""You are a security-camera episodic-memory extractor. You \
watch a short clip from ONE fixed camera and output ONE JSON object describing what \
is in view and what happened. Your output is stored so the owner can later ask \
questions like "who took the orange car and when".

Return ONLY the JSON object — no markdown, no code fences, no commentary.

Schema (all fields required):
{{
  "activity_type": use "watching",
  "event_type": one of [start, progress, switch, complete, idle, other],
  "project": null,
  "summary": a concise factual paragraph of what the camera sees and what changed,
  "importance": float 0.0-1.0 (how notable this is; a person or vehicle appearing/\
leaving is high, an empty static scene is low),
  "confidence": float 0.0-1.0,
  "entities": [{{"name": a specific description ("orange car", "man in blue jacket", \
"delivery van"), "type": a lowercase noun (prefer one of [{", ".join(_CAMERA_ENTITY_TYPES)}]), \
"confidence": 0.0-1.0}}],
  "claims": [{{"text": a factual EVENT you can support from the clip ("the orange car \
was driven out of the garage", "a person walked to the front door and left a package"), \
"confidence": 0.0-1.0}}],
  "tasks": [],
  "boundary_signal": one of [continuation, new_event, boundary]
}}

Rules:
- Name people and vehicles concretely by their visible appearance (color, type, \
clothing) — never vague tokens like "object" or "thing" when you can be specific.
- claims are the timeline of what HAPPENED (arrivals, departures, someone taking or \
moving something). If nothing changed, use an empty list [].
- If the scene is empty/static, say so in summary, importance low, entities/claims [].
- summary must always be present and non-empty.
- project is always null.
"""

_CAMERA_USER_PROMPT = (
    "Extract the structured JSON record for this camera clip. Output only the JSON object."
)


class CameraCaptureWorker:
    """Owns one RTSP stream + its own MemoryPipeline, room, and worker thread."""

    def __init__(self, camera_id, name, rtsp_url, model_name_vlm,
                 neo4j_store=None, activity_logger=None,
                 window_seconds=60, fps=1.0, notification_sink=None,
                 insight_callback=None):
        self.camera_id = camera_id
        self.name = name
        self.rtsp_url = rtsp_url
        self.model_name_vlm = model_name_vlm
        self.neo4j = neo4j_store
        self.window_seconds = int(window_seconds)
        self.fps = float(fps)
        self.insight_callback = insight_callback

        # Keep a little more than one window of frames so a batch is never starved.
        self.stream = RealtimeCameraStream(
            rtsp_url, window_size=int(window_seconds * fps) + 5, fps=fps)
        # Per-camera pipeline: independent SessionManager so this camera's timeline
        # never entangles with the screen's or another camera's.
        self.pipeline = MemoryPipeline(
            id_strategy="deterministic", expected_seconds=window_seconds,
            neo4j_store=neo4j_store, activity_logger=activity_logger, jsonl=False,
            log_context="camera", notification_sink=notification_sink)

        # Own VLM client so this worker thread never shares an httpx pool with
        # the screen worker or another camera (see providers.local_openAI).
        self._client = new_vlm_client()
        self._paused = False
        self._stop = Event()
        self.frames_processed = 0
        self.events_logged = 0
        self.last_processed_at = None
        self.last_summary = None
        self.last_error = None

        # Motion gate: only spend a VLM call when the scene actually changed.
        # Adaptive background subtraction absorbs wind/foliage; see MotionGate.
        self.gate = MotionGate(
            min_area_frac=float(os.getenv("CAMERA_MOTION_MIN_AREA", "0.008")),
            min_motion_frames=int(os.getenv("CAMERA_MOTION_MIN_FRAMES", "3")),
            var_threshold=float(os.getenv("CAMERA_MOTION_SENSITIVITY", "30")),
        )
        # Heartbeat: record the scene every N idle windows even without motion, so
        # a slowly-changed static view is still remembered. 0 disables.
        self.heartbeat_windows = int(os.getenv("CAMERA_HEARTBEAT_MINUTES", "15"))
        self._idle_windows = 0
        self.last_motion = None
        # Max tokens for the VLM extraction. The screen path uses 2500; cameras
        # need room for the summary plus a full entity/claim list.
        self._max_tokens = int(os.getenv("CAMERA_MAX_TOKENS", "5000"))

        self._thread = Thread(target=self._run, name=f"camera:{camera_id}", daemon=True)
        self._thread.start()

    # -- control -----------------------------------------------------------
    def pause(self):
        self._paused = True
        logger.info("Camera %s paused.", self.camera_id)

    def resume(self):
        self._paused = False
        logger.info("Camera %s resumed.", self.camera_id)

    @property
    def paused(self):
        return self._paused

    # -- worker loop -------------------------------------------------------
    def _run(self):
        # Wait for the RTSP reader to warm up before the first tick.
        while not self._stop.is_set() and not self.stream.healthy and self.stream.running:
            self._stop.wait(1.0)
        while not self._stop.is_set():
            self._stop.wait(self.window_seconds)
            if self._stop.is_set():
                break
            if self._paused or not self.stream.running:
                continue
            try:
                self._process_once()
            except Exception as exc:
                self.last_error = str(exc)
                logger.exception("camera %s processing failed: %s", self.camera_id, exc)

    def _process_once(self):
        pil_frames = self.stream.frames()
        if len(pil_frames) < 2:
            return
        frames = [np.asarray(f.convert("RGB")) for f in pil_frames]
        timestamp = time.time()

        # Gate on real motion. The gate still learns the scene every window; we
        # only skip the (expensive) VLM call + event when nothing moved — with a
        # periodic heartbeat so a static view is still remembered occasionally.
        moved, motion = self.gate.evaluate(frames)
        self.last_motion = motion
        if not moved:
            self._idle_windows += 1
            heartbeat = (self.heartbeat_windows > 0
                         and self._idle_windows >= self.heartbeat_windows)
            if not heartbeat:
                logger.debug("Camera %s: no motion (%s) — skipping VLM.",
                             self.camera_id, motion)
                return
            logger.info("Camera %s: heartbeat capture after %d idle window(s).",
                        self.camera_id, self._idle_windows)
        self._idle_windows = 0

        result, status = asyncio.run(self._extract(frames))
        full_summary = (result.summary or "").strip()
        # Short preview for the UI status line (the full text is stored in memory).
        self.last_summary = full_summary[:200]
        logger.info("Camera %s extraction (%s, %d chars, %d entities, %d claims): %s",
                    self.camera_id, status, len(full_summary),
                    len(result.entities), len(result.claims), full_summary)

        # `application` becomes the event's source tag in the Cameras room, so it
        # carries the camera's display name ("IPC-A22E-G") rather than its id —
        # all cameras share one room now, and the tag is what tells them apart.
        # The summary as the title makes the graph Event read meaningfully.
        batch = {
            "timestamp": timestamp,
            "window_titles": [self.last_summary or self.name],
            "process_names": [self.name or self.camera_id],
            "repr_frame": frames[-1],
            "extraction": {**result.model_dump(), "selected_profile": "camera"},
        }
        self.pipeline.ingest(batch)
        if self.insight_callback is not None and full_summary:
            try:
                self.insight_callback(
                    full_summary, timestamp, f"camera:{self.name}",
                    {"camera_id": self.camera_id, "camera_name": self.name,
                     "motion": motion})
            except Exception as exc:
                logger.warning("Camera %s proactive callback failed: %s",
                               self.camera_id, exc)
        self.frames_processed += len(frames)
        self.events_logged += 1
        self.last_processed_at = timestamp

    async def _extract(self, frames):
        video_b64 = _encode_frames_to_mp4_base64(frames, fps=self.fps)
        base_content = [
            {"type": "text", "text": _CAMERA_USER_PROMPT},
            {"type": "video_url",
             "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
        ]

        async def generate(feedback):
            user_content = list(base_content)
            if feedback:
                user_content = [{"type": "text", "text": feedback}] + user_content
            messages = [
                {"role": "system", "content": _CAMERA_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            resp = await self._client.chat.completions.create(
                model=self.model_name_vlm, messages=messages, max_tokens=self._max_tokens)
            return resp.choices[0].dict()["message"]["content"]

        return await run_extraction(generate)

    # -- status / teardown -------------------------------------------------
    def status(self):
        st = self.stream.status()
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "configured": True,
            "connected": bool(st.get("healthy")),
            "healthy": bool(st.get("healthy")) and not self._paused,
            "paused": self._paused,
            "buffered_frames": st.get("frames", 0),
            "events_logged": self.events_logged,
            "last_processed_at": self.last_processed_at,
            "last_summary": self.last_summary,
            "last_motion": self.last_motion,
            "error": self.last_error or st.get("error"),
        }

    def cleanup(self):
        self._stop.set()
        try:
            self.stream.cleanup()
        except Exception:
            logger.warning("camera %s stream cleanup failed", self.camera_id, exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=10)
        try:
            self.pipeline.finalize()
        except Exception as exc:
            logger.warning("camera %s pipeline.finalize failed: %s", self.camera_id, exc)


def _encode_frames_to_mp4_base64(frames, fps=1.0):
    """Encode a list of RGB numpy frames to a base64 MP4 (mirrors screen.py)."""
    if not frames:
        return None
    height, width = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_filename = f.name
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_filename, fourcc, max(fps, 1.0), (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        out.release()
        with open(temp_filename, "rb") as vf:
            return base64.b64encode(vf.read()).decode("utf-8")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

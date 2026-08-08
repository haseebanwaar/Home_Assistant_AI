"""Camera capture worker — one live RTSP camera → episodic memory.

Unlike the screen path (which keys sessions on app/window boundaries), a camera
has no "application" to switch between. So a worker treats the camera itself as
the session: every `window_seconds` it grabs the recent frames, asks the VLM for
a surveillance-focused structured extraction (people, vehicles, objects, and the
*events* between them — "the orange car was driven out of the garage"), and feeds
that through the shared MemoryPipeline. That writes an Event + entities/claims to
Neo4j and Qdrant exactly like the screen path, and routes the event into this
camera's own Room so its feed reads like a channel of what the camera saw.

Windows are not independent, though. Each one is extracted against the camera's
persistent scene — the slots it is already tracking, with how long each has been
in its current state — and folded back into it afterwards (see
sources.camera_state). That changes what a clip is worth saying: a window where
every slot is exactly as it was is a *confirmation*, counted and used to extend
the duration but never written to the room, and a window where something moved
is narrated against what came before ("the black gate opened after 7h 20m; it
usually opens around 09:00"). Without that, every second clip said "the gate is
closed" and the one clip that mattered was buried under them.

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
from collections import defaultdict, deque
from threading import Event, Thread

import cv2
import numpy as np

from sources.rtsp import RealtimeCameraStream
from sources.video_writer import open_mp4_writer
from sources.frame_budget import frames_as_image_parts
from sources.motion_gate import MotionGate
from sources.camera_validation import HEALTH, classify
from sources.camera_state import is_infrared
from sources.capture_settings import (
    expected_frame_count,
    validate_capture_profile,
)
from memory.models.extraction import ENTITY_TYPE_SUGGESTIONS
from memory.extraction.validator import run_extraction
from memory.pipeline import MemoryPipeline
from providers.local_openAI import new_vlm_client, thinking_request_kwargs
from utils.maintenance import maintenance_window_active

logger = logging.getLogger("home_assistant")


# Entity vocabulary that matters for a physical scene, layered on the shared
# suggestions so the VLM still tags anything sensible it sees.
_CAMERA_ENTITY_TYPES = [
    "person", "vehicle", "car", "truck", "motorcycle", "bicycle",
    "animal", "package", "door", "bag", "tool", "object",
] + ENTITY_TYPE_SUGGESTIONS

_CAMERA_SYSTEM_PROMPT = f"""You are a fixed-camera episodic-memory extractor. You \
watch a chronological clip from ONE fixed camera and output ONE JSON object that \
preserves a continuous, queryable account of the property. The owner must later be \
able to ask questions such as "describe the person who took the red car", "how many \
cars were parked", and "who entered the house and how many people came in".

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
  "states": [{{"key": the existing slot key from "Persistent scene state" when this \
is that same physical thing, otherwise null, "subject": what it is ("black gate", \
"orange car"), "state": the condition it is in AT THE END of this clip, as one short \
phrase ("closed", "parked in the driveway facing out"), "confidence": 0.0-1.0}}],
  "gone": [slot keys from the tracked list that this clip shows are genuinely no \
longer there],
  "boundary_signal": one of [continuation, new_event, boundary]
}}

Rules:
- Treat the images as ordered samples from one continuous clip. Reconstruct the \
sequence from earliest to latest; do not describe them as unrelated screenshots.
- Focus on observable property activity and scene state, not camera/stream health. \
Record people, entrances/exits, gate movement, vehicles arriving/leaving, people \
interacting with vehicles, carried items, and useful before/after state.
- Make every record independently useful later: state visible person/vehicle \
descriptions, direction of travel, relevant location, and counts when supportable.
- Put every relevant visible person and vehicle in entities, even when stationary. \
Do not list permanent background fixtures or ordinary plant movement as entities.
- Maintain identity continuity within the clip using stable appearance descriptions. \
Never identify a person by name unless the supplied context establishes the name.
- Distinguish observed facts from uncertainty. Never invent an arrival, departure, \
entry, exit, ownership, identity, or causal link that the clip does not show.
- For vehicles, include color, type, distinguishing features and parking position; \
for people, include clothing colors, apparent build and carried items when visible.
- When someone interacts with, enters, exits, drives, or walks away with a vehicle, \
connect that same appearance description to the vehicle in summary and claims.
- Name people and vehicles concretely by their visible appearance (color, type, \
clothing) — never vague tokens like "object" or "thing" when you can be specific.
- claims are the timeline of what HAPPENED (arrivals, departures, someone taking or \
moving something). If nothing changed, use an empty list [].
- states is the opposite of claims: what is STILL TRUE when the clip ends. List every \
tracked slot you can still see plus anything else standing that is worth following \
(gates, doors, parked vehicles, objects left out). It is normal for states to be full \
while claims is empty — that is a quiet scene, not an empty one.
- The persistent slots you are given already are the memory of this camera across \
hours and days. Reuse a slot's key for the same physical thing and echo its stored \
state text back exactly when nothing about it changed. Opening a second slot for \
something already tracked destroys the history of how long it has been that way.
- If the scene is static, summarize the current useful inventory (including counts \
and locations of people/vehicles) with low importance and claims [].
- Do not turn image corruption, blur, or connectivity into the main observation. If \
the scene cannot be interpreted reliably, say that briefly and lower confidence.
- summary must always be present and non-empty.
- project is always null.
"""

_CAMERA_SCENE_CONTEXTS = {
    "ipc-a42-l": """This camera is IPC-A42-L, fixed above the black gate. The center \
of the image is the black gate; the left side looks inward toward the garage/home; \
the right side looks outward toward the road. Prioritize gate opening/closing and \
people waiting or standing outside on the road/right side. Track whether people and \
vehicles move from outside/right through the gate toward inside/left or the reverse. \
Do not confuse merely standing outside with entering the property.""",
    "ipc-s42-f": """This camera is IPC-S42-F, fixed in the open garage. The black \
gate, parked cars, and garden plants are visible. Keep an explicit inventory and \
count of parked cars, with colors/types/positions. Prioritize cars arriving, leaving, \
or changing position and connect visible people to those vehicle interactions. Plant \
movement alone is background motion, not a property event, but meaningful changes to \
the plants or a person tending them should be recorded.""",
}


def camera_scene_context(name):
    """Return stable layout knowledge for a specifically positioned fixed camera."""
    return _CAMERA_SCENE_CONTEXTS.get(str(name or "").strip().lower(), "")


_INFRARED_NOTE = """This clip is in infrared/night mode: the picture has no real \
colour, so everything reads as white, grey or black. Do NOT name or re-name anything \
by the colour it appears here, and do not open a new slot because a tracked thing \
looks a different colour than it did in daylight — match it by shape, size and \
position and keep the name it already has. Describe colour only as unknown under \
night vision."""


def camera_user_prompt(name, previous_observation=None, state_block="",
                       infrared=False):
    """Build per-window instructions with scene layout and timeline continuity."""
    parts = [
        f"Camera: {name or 'unknown fixed camera'}.",
        camera_scene_context(name),
    ]
    if infrared:
        parts.append(_INFRARED_NOTE)
    if state_block:
        parts.append(state_block)
    if previous_observation:
        parts.append(
            "Previous stored observation (continuity context, not proof of a new "
            f"event): {previous_observation}"
        )
        parts.append(
            "Compare the current clip with that prior state when useful, but only "
            "claim a transition when the current visual evidence supports it."
        )
    parts.append(
        "Extract the structured JSON record for this chronological camera clip. "
        "Output only the JSON object."
    )
    return "\n\n".join(part for part in parts if part)


class CameraCaptureWorker:
    """Owns one RTSP stream + its own MemoryPipeline, room, and worker thread."""

    def __init__(self, camera_id, name, rtsp_url, model_name_vlm,
                 neo4j_store=None, activity_logger=None,
                 window_seconds=120, fps=0.5, notification_sink=None,
                 insight_callback=None, clip_store=None, state_store=None):
        self.camera_id = camera_id
        self.name = name
        self.rtsp_url = rtsp_url
        self.model_name_vlm = model_name_vlm
        self.neo4j = neo4j_store
        self.fps, self.window_seconds = validate_capture_profile(
            fps, window_seconds
        )
        self.insight_callback = insight_callback
        # Evidence clip for this window. Written only for windows we keep, so an
        # alert or a nudge about this camera can be watched and asked about.
        self.clip_store = clip_store
        # The scene between the clips: what this camera is tracking, since when,
        # and what it usually does. See sources.camera_state.
        self.state_store = state_store

        # Keep a little more than one window of frames so a batch is never starved.
        self.stream = RealtimeCameraStream(
            rtsp_url,
            window_size=expected_frame_count(self.fps, self.window_seconds),
            fps=self.fps,
        )
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
        self._profile_changed = Event()
        self.frames_processed = 0
        self.events_logged = 0
        self.last_processed_at = None
        self.last_summary = None
        # Last observation actually committed to memory. Passing it into the next
        # extraction preserves scene continuity without promoting rejected health
        # diagnostics into evidence.
        self._previous_observation = None
        self.last_error = None
        # Windows where every tracked slot was found exactly as it was. Not an
        # error and not a health problem — the camera did its job and the scene
        # simply had not moved — so they are counted apart from both.
        self.steady_windows = 0
        self.last_continuity = []
        self.last_infrared = None
        # Extractions that were not worth remembering. Kept (bounded) rather than
        # dropped on the floor: a camera whose feed has degraded shows up here as
        # a rising 'picture distorted' rate long before anyone notices the room
        # has gone quiet.
        self.health_records = deque(maxlen=50)
        self.discarded = 0
        self.discarded_by_reason = defaultdict(int)

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
        self._min_confidence = float(os.getenv("CAMERA_MIN_CONFIDENCE", "0.35"))

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
            profile_changed = self._profile_changed.wait(self.window_seconds)
            self._profile_changed.clear()
            if self._stop.is_set():
                break
            if profile_changed:
                continue
            if self._paused or not self.stream.running:
                continue
            try:
                self._process_once()
            except Exception as exc:
                self.last_error = str(exc)
                logger.exception("camera %s processing failed: %s", self.camera_id, exc)

    def _process_once(self):
        if maintenance_window_active():
            logger.debug("Camera %s: maintenance window, skipping inference.",
                         self.camera_id)
            return
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
        heartbeat = False
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

        # Night vision kills colour, and colour is how the extractor recognises
        # a car it is already tracking. Detect it from the pixels and say so in
        # the prompt rather than letting the orange car become a white one.
        infrared = is_infrared(frames)
        self.last_infrared = infrared
        result, status = asyncio.run(self._extract(frames, infrared))
        full_summary = (result.summary or "").strip()
        # Short preview for the UI status line (the full text is stored in memory).
        self.last_summary = full_summary[:200]
        logger.info("Camera %s extraction (%s, %d chars, %d entities, %d claims): %s",
                    self.camera_id, status, len(full_summary),
                    len(result.entities), len(result.claims), full_summary)

        # Motion opened the gate; this decides whether the result is worth
        # remembering. A decode glitch moves most of the frame and so passes the
        # gate exactly like a person would — only the description reveals there
        # was nothing there. See sources.camera_validation.
        verdict, reason = classify(
            full_summary, entities=[e.name for e in result.entities],
            claims=[c.text for c in result.claims],
            confidence=result.confidence, heartbeat=heartbeat,
            min_confidence=self._min_confidence)
        if verdict == HEALTH:
            # Deliberately before the state update: a broken picture must never
            # be allowed to rewrite what the camera believes about the scene.
            self._record_health(reason, full_summary, timestamp)
            logger.info("Camera %s: extraction not stored (%s).",
                        self.camera_id, reason)
            self.frames_processed += len(frames)
            self.last_processed_at = timestamp
            return

        # Fold this window into the standing scene, then say what moved. The
        # continuity lines are the difference between "the gate is closed" and
        # "the gate has not opened since 06:12 this morning".
        delta, continuity = self._update_state(result, timestamp)
        self.last_continuity = continuity

        # A window that found every tracked slot exactly as it was is a state
        # confirmation, not an event. Storing those is what made the room read
        # as the same sentence every two minutes; the durations they extend are
        # kept, and the periodic heartbeat still records an idle scene on
        # purpose so a quiet camera is not invisible.
        if delta is not None and delta.confirmed_only and not heartbeat:
            self.steady_windows += 1
            self._previous_observation = full_summary
            logger.info("Camera %s: scene unchanged (%d slot(s) confirmed) — %s",
                        self.camera_id, len(delta.unchanged),
                        "; ".join(continuity) or "no change")
            self.frames_processed += len(frames)
            self.last_processed_at = timestamp
            return

        # What gets remembered is the observation plus its continuity, so the
        # clip read back tomorrow still carries how long things had been that
        # way. The raw summary stays the continuity context for the next window
        # — feeding computed durations back in would let them compound.
        stored_summary = full_summary
        if continuity:
            stored_summary = (full_summary + "\n\nContinuity: "
                              + "; ".join(continuity) + ".")

        # `application` becomes the event's source tag in the Cameras room, so it
        # carries the camera's display name ("IPC-A22E-G") rather than its id —
        # all cameras share one room now, and the tag is what tells them apart.
        # The summary as the title makes the graph Event read meaningfully.
        # Record the clip BEFORE ingest: the notification sink fires inside
        # ingest, so the clip id has to already exist for an alert to carry it.
        clip_id = None
        if self.clip_store is not None:
            clip_id = self.clip_store.save(
                frames, source="camera", label=self.name or self.camera_id,
                timestamp=timestamp, capture_fps=self.fps, summary=stored_summary,
                extra={"camera_id": self.camera_id, "camera_name": self.name})
        # Hang the footage on the transitions this window recorded, so "when did
        # the gate open" can be answered with the clip of it opening.
        if clip_id and self.state_store is not None and delta is not None:
            try:
                self.state_store.attach_clip(self.camera_id, timestamp, clip_id)
            except Exception as exc:
                logger.debug("Camera %s: could not attach clip to transitions: %s",
                             self.camera_id, exc)

        batch = {
            "timestamp": timestamp,
            "window_titles": [self.last_summary or self.name],
            "process_names": [self.name or self.camera_id],
            "repr_frame": frames[-1],
            "clip_id": clip_id,
            "extraction": {**result.model_dump(), "summary": stored_summary,
                           "selected_profile": "camera"},
        }
        ingested = self.pipeline.ingest(batch)
        self._previous_observation = full_summary
        # Now the window has an event id, so the clip can be found from the
        # timeline as well as from the alert that referenced it.
        if clip_id and self.clip_store is not None:
            try:
                self.clip_store.annotate(
                    clip_id, event_id=ingested.current_event.event_id)
            except Exception as exc:
                logger.debug("Could not annotate clip %s: %s", clip_id, exc)
        if self.insight_callback is not None and full_summary:
            try:
                self.insight_callback(
                    stored_summary, timestamp, f"camera:{self.name}",
                    {"camera_id": self.camera_id, "camera_name": self.name,
                     "motion": motion, "clip_id": clip_id,
                     "infrared": infrared, "continuity": continuity,
                     "state": delta.as_dict() if delta is not None else None})
            except Exception as exc:
                logger.warning("Camera %s proactive callback failed: %s",
                               self.camera_id, exc)
        self.frames_processed += len(frames)
        self.events_logged += 1
        self.last_processed_at = timestamp

    def _update_state(self, result, timestamp):
        """Fold the extraction into this camera's standing scene.

        Returns (delta, continuity lines). A window the extractor gave no states
        for yields `None` rather than an empty delta: "reported nothing" and
        "reported that nothing changed" are different claims, and only the second
        one may be used to suppress an event.
        """
        if self.state_store is None or not getattr(result, "states", None):
            return None, []
        try:
            delta = self.state_store.apply(
                self.camera_id, result.states, gone=getattr(result, "gone", ()),
                timestamp=timestamp)
            return delta, self.state_store.continuity_lines(
                self.camera_id, delta, now=timestamp)
        except Exception as exc:
            # State tracking is an enrichment; losing it must not lose the event.
            logger.warning("Camera %s state tracking failed: %s",
                           self.camera_id, exc, exc_info=True)
            return None, []

    def _tracked_count(self):
        """How many slots this camera is carrying — never fails a status poll."""
        if self.state_store is None:
            return 0
        try:
            return len(self.state_store.slots(self.camera_id))
        except Exception:
            return 0

    def state(self):
        """Everything this camera currently believes is true of its scene."""
        if self.state_store is None:
            return {"camera_id": self.camera_id, "name": self.name,
                    "enabled": False, "states": [], "rhythms": [],
                    "overdue": [], "recent_transitions": []}
        return {**self.state_store.snapshot(self.camera_id), "name": self.name}

    async def _extract(self, frames, infrared=False):
        image_parts, frame_info = frames_as_image_parts(frames)
        if not image_parts:
            raise ValueError("could not prepare camera frames for inference")
        state_block = ""
        if self.state_store is not None:
            try:
                state_block = self.state_store.prompt_block(self.camera_id)
            except Exception as exc:
                logger.warning("Camera %s could not load tracked state: %s",
                               self.camera_id, exc)
        base_content = [
            {"type": "text", "text": camera_user_prompt(
                self.name, self._previous_observation, state_block=state_block,
                infrared=infrared)},
            *image_parts,
        ]
        logger.info("Camera %s inference using %d/%d temporal images at %sx%s",
                    self.camera_id, frame_info["kept"], len(frames),
                    frame_info["width"], frame_info["height"])

        async def generate(feedback):
            user_content = list(base_content)
            if feedback:
                user_content = [{"type": "text", "text": feedback}] + user_content
            messages = [
                {"role": "system", "content": _CAMERA_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            resp = await self._client.chat.completions.create(
                job_label=f"Camera extraction — {self.name or self.camera_id}",
                model=self.model_name_vlm, messages=messages,
                max_tokens=self._max_tokens,
                **thinking_request_kwargs(False),
            )
            return resp.choices[0].dict()["message"]["content"]

        return await run_extraction(generate)

    def _record_health(self, reason, summary, timestamp):
        self.discarded += 1
        self.discarded_by_reason[reason] += 1
        self.health_records.append({
            "timestamp": timestamp, "reason": reason,
            "summary": (summary or "")[:300],
        })

    def health(self):
        """Why this camera's windows were not stored, newest first."""
        attempted = self.events_logged + self.discarded
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "stored": self.events_logged,
            "discarded": self.discarded,
            # Not a discard: the scene was read successfully and had not moved.
            "steady_windows": self.steady_windows,
            "discard_pct": (round(100.0 * self.discarded / attempted, 1)
                            if attempted else None),
            "by_reason": dict(self.discarded_by_reason),
            "flat_frame_pct": self.stream.status().get("flat_frame_pct"),
            "recent": list(self.health_records)[::-1],
        }

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
            "last_frame_at": st.get("last_frame_at"),
            "reconnects": st.get("reconnects", 0),
            "flat_frame_pct": st.get("flat_frame_pct"),
            "last_frame_std": st.get("last_frame_std"),
            "events_logged": self.events_logged,
            "discarded": self.discarded,
            "last_discard_reason": (self.health_records[-1]["reason"]
                                    if self.health_records else None),
            "last_processed_at": self.last_processed_at,
            "last_summary": self.last_summary,
            "last_motion": self.last_motion,
            "steady_windows": self.steady_windows,
            "tracked_states": self._tracked_count(),
            "continuity": self.last_continuity,
            "infrared": self.last_infrared,
            "sample_fps": self.fps,
            "inference_interval_seconds": self.window_seconds,
            "thinking": False,
            "expected_frames": expected_frame_count(
                self.fps, self.window_seconds
            ),
            "error": self.last_error or st.get("error"),
        }

    def update_capture_profile(self, sample_fps, inference_interval_seconds):
        """Start a fresh sampling/inference window using the new profile."""
        fps, interval = validate_capture_profile(
            sample_fps, inference_interval_seconds
        )
        self.fps = fps
        self.window_seconds = interval
        self.stream.update_sampling(
            fps, expected_frame_count(fps, interval)
        )
        self.pipeline.expected_seconds = interval
        self._idle_windows = 0
        self._profile_changed.set()
        logger.info(
            "Camera %s capture profile changed to %.3g fps / %ds; "
            "frame buffer reset.",
            self.camera_id,
            fps,
            interval,
        )

    def cleanup(self):
        self._stop.set()
        self._profile_changed.set()
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
        out = open_mp4_writer(temp_filename, fps, (width, height))
        if out is None:
            logger.warning("Could not open a video writer for camera frames.")
            return None
        for frame in frames:
            out.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        out.release()
        with open(temp_filename, "rb") as vf:
            return base64.b64encode(vf.read()).decode("utf-8")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

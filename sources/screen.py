import base64
import logging
import math
import os
import time
import asyncio
from collections import deque
from threading import Thread, Lock, Event
import cv2
from PIL import Image
from mss import mss
import numpy as np
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from providers.local_openAI import client, get_model_name_vlm, thinking_request_kwargs
from utils.qwen_preprocess import encode_video
import cv2
import tempfile
import pygetwindow as gw

from memory.models.observation import Observation
from memory.debug import write_jsonl
from sources.capture_settings import (
    expected_frame_count,
    validate_capture_profile,
)
from sources.idle import PresenceGate
from sources.video_writer import open_mp4_writer
from sources.frame_budget import frames_as_image_parts

# Windows-only PID lookup for the active window (Step 1: process_name capture).
try:
    import psutil
    import win32process
except Exception:  # pragma: no cover - non-Windows / missing deps
    psutil = None
    win32process = None

logger = logging.getLogger("home_assistant")


def _process_for_hwnd(hwnd):
    """Return (process_name, application) for a window handle, or (None, None).

    process_name is the executable (e.g. "opera.exe"); application is the
    friendly stem (e.g. "opera").
    """
    if not hwnd or win32process is None or psutil is None:
        return None, None
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        name = psutil.Process(pid).name()
        stem = os.path.splitext(name)[0].lower()
        return name, stem
    except Exception as exc:
        logger.debug("process lookup failed for hwnd %s: %s", hwnd, exc)
        return None, None


class RealtimeScreenCapture:
    def __init__(self, video_source,model_name_vlm, window_size=60, fps=1.0, monitor_index=1, target_resolution=None,
                 activity_logger=None, insight_callback=None, start_capture=True, pipeline=None,
                 clip_store=None, presence_gate=None, thinking=True):
        """
        Args:
            video_source: screen (not used)
            window_size: Number of seconds to keep in memory
            fps: Frames per second to process
            monitor_index: the index of the monitor to capture (default 1 for primary)
            target_resolution: a tuple of (width, height) for resizing, or None to keep original resolution
            activity_logger: an instance of ActivityLogger to log each minute of activity
            presence_gate: a PresenceGate; minutes where the user is neither at
                the keyboard nor watching a repainting screen are not captured
        """
        self.video_source = video_source
        self.fps, self.window_size = validate_capture_profile(
            fps, window_size
        )
        self.frame_buffer = deque(
            maxlen=expected_frame_count(self.fps, self.window_size)
        )
        self.lock = Lock()
        self._profile_changed = Event()
        self._profile_version = 0
        self.running = True
        # When paused, the capture loop keeps the thread alive but stops grabbing
        # frames and processing batches, so screen memory can be halted from the UI
        # without tearing down the stream. Resume picks straight back up.
        self.paused = False
        self.healthy = False
        self.last_error = None
        self.monitor_index = monitor_index
        self.model_name_vlm = model_name_vlm
        self.thinking = bool(thinking)
        self.max_tokens = int(os.getenv("SCREEN_MAX_TOKENS", "18000"))
        self.target_resolution = target_resolution
        self.activity_logger = activity_logger
        # Optional callback(description, timestamp) invoked after each minute is
        # described — used for the proactive path. Runs in the describe thread.
        self.insight_callback = insight_callback
        # Low-res evidence clip per processed minute, so a nudge or an alert
        # about what was on screen can be replayed and questioned.
        self.clip_store = clip_store
        self.current_minute_apps = list()
        # Step 3: process names seen this minute, used to route the domain profile.
        self.current_minute_processes = list()
        # Step 1: last (window_title, process_name) written, so we only emit a
        # structured Observation when the active window actually changes.
        self._last_observed_key = None
        # Step 2: when on, the VLM emits validated JSON (ExtractionResult) and we
        # log result.summary to Qdrant (app unchanged) + the full object to
        # data/debug/extractions.jsonl.
        self.structured_extraction = os.getenv("STRUCTURED_EXTRACTION", "0").lower() in ("1", "true", "yes")
        # Step 4: gap-spanning frame sampling before the VLM call.
        self.gap_sampling = os.getenv("GAP_SAMPLING", "0").lower() in ("1", "true", "yes")
        self.gap_sample_count = int(os.getenv("GAP_SAMPLE_COUNT", "10"))
        # Heartbeat: force a capture after this many idle (low-activity) minutes so
        # static reading/watching is still remembered. 0 disables. Default 3.
        self.heartbeat_minutes = int(os.getenv("HEARTBEAT_MINUTES", "3"))
        self._idle_minutes = 0
        # Is the user there? Keyboard/mouse says yes for anything interactive; a
        # large share of the screen repainting says yes for watching, which has no
        # input at all (sources.idle.PresenceGate). Minutes with neither are
        # dropped entirely: nothing captured, nothing timed, nothing to attribute
        # to whichever window happened to be in front.
        self.presence_gate = (presence_gate if presence_gate is not None
                              else PresenceGate())
        self._away_minutes = 0
        # Per-second repaint samples for the current window, consumed by the gate.
        self._minute_changes = []
        # Only record the active window if its center is inside the captured
        # monitor. Off by default: on multi-monitor / DPI-scaled setups the
        # coordinate systems mismatch and this drops valid titles, collapsing
        # sessions. The focused window is the signal regardless of monitor math.
        self._strict_monitor_bounds = os.getenv("STRICT_MONITOR_BOUNDS", "0").lower() in ("1", "true", "yes")
        self.describe_thread = None # Thread for describing frames
        self.describe_thread_lock = Lock()  # Lock for the thread
        # self.description_history = deque(maxlen=1)  #store last 3 description for context

        # Step-a: live memory pipeline (sessions/events/knowledge, dual store).
        # When set, batches feed pipeline.ingest() and per-minute log_activity is
        # replaced by the pipeline's event-scoped logging.
        self.pipeline = pipeline
        # Single-worker + 1-slot COALESCING mailbox: if the worker is still busy
        # when a new minute lands, the pending batch is replaced by the newer one
        # (memory reflects "now"; lag can't accumulate). Dropped batches counted.
        self._need_process = self.activity_logger is not None or self.pipeline is not None
        self._mailbox = None
        self._mailbox_lock = Lock()
        self._mailbox_wake = Event()
        self._dropped_batches = 0
        self._worker_thread = None

        # Step 0: optional capture recorder for the offline replay harness.
        self.recorder = None
        if os.getenv("RECORD_CAPTURE", "0").lower() in ("1", "true", "yes"):
            from sources.capture_recorder import CaptureRecorder
            self.recorder = CaptureRecorder()
            logger.info("RECORD_CAPTURE enabled — dumping batches to %s", self.recorder.run_dir)
        else:
            logger.info("RECORD_CAPTURE disabled (set RECORD_CAPTURE=1 to record for replay).")

        # Start the single processing worker (drains the coalescing mailbox).
        if start_capture and self._need_process:
            self._worker_thread = Thread(target=self._batch_worker, daemon=True)
            self._worker_thread.start()

        # Start frame capture thread (skipped by the replay harness).
        self.capture_thread = None
        if start_capture:
            self.capture_thread = Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()

    async def _describe_frames(self, imgs):
        """
        Describe the current minute of activity, using VLM
        """
        if len(imgs) == 0:
            return ''
        image_parts, frame_info = frames_as_image_parts(imgs)
        if not image_parts:
            return "Error preparing screen frames"

        # also provie ;long term memory here. like for an hour? b
        # if len(self.description_history) > 10:
        #     question += f'Relevant past context:\n'
        #     for i, description in enumerate(self.description_history):
        #         question += f'{description}\n'
        question = "describe what you see on user PC so that if i read your description later i will get full meaning ?\n"
        question = """Describe exactly what is happening on the screen right now.
Include:
– what the user is doing or experiencing
– important visible text, subtitles, or dialogue
– actions or story events
– what is visually changing and why it matters

Focus on information useful to remember later.
Someone reading your description should be able to continue watching the experience without missing anything."""
        sys_prompt =  """
            You are a visual narrator describing exactly what appears on the user's PC screen.

Your job:
- you are provided with user screen captured every minute. dont miss any detail.if there are subtitles, make use of that, if there is text do read it all.
- Give a clear, faithful description of what is visible in the video.
- Be detailed enough that someone who cannot see the screen could follow what is happening.
                    """

        sys_prompt = """You are a visual episodic memory recorder.
You observe the user’s computer screen and produce a concise, factual timeline of what is happening.
Your output will be used for memory retrieval later, so focus on meaning and key changes over time.

Follow these rules:
1️⃣ Describe the main activity or purpose (e.g., watching a video, coding, browsing, gaming)
2️⃣ Include important on-screen text, subtitles, titles, and readable UI labels
3️⃣ Identify people or characters and what they are doing only if relevant
4️⃣ Describe scene changes, interactions, and visible progress indicators
5️⃣ Capture semantic content — topics, story beats, intentions, goals
6️⃣ Omit irrelevant sensory details (clothes, wall colors) unless meaningful
7️⃣ Write in clear paragraph format, no lists or bullet points
8️⃣ Avoid guessing — if uncertain, state what is likely based only on visuals

Your goal:
Create a retrievable memory record that preserves what matters most for future recall"""


        messages = [
            {"role": "system", "content": f"{sys_prompt}"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question}"},
                    *image_parts,
                ]
            }
        ]
        tim = time.perf_counter()
        response = await client.chat.completions.create(
            model=self.model_name_vlm, messages=messages,
            max_tokens=self.max_tokens if self.thinking else 2500,
            **thinking_request_kwargs(self.thinking),
        )

        answer = response.choices[0].dict()['message']['content']
        logger.debug("Screen description: %s", answer)
        logger.info("Screen processing of %d frames took %.3f seconds", len(imgs), time.perf_counter()-tim)


        return answer

    async def _extract_structured(self, imgs, process_names=None, window_titles=None):
        """Step 2: run the VLM and return (ExtractionResult, status, profile_name).

        status is "ok" | "retry" | "fallback" | "empty". Shares the video-encode
        and VLM-call machinery with the prose path so live and replay agree.
        process_names/window_titles default to the current-minute state (used by
        replay, which sets it); the live worker passes an explicit snapshot.
        """
        from memory.models.extraction import ExtractionResult
        from memory.extraction.prompts import (
            build_system_prompt, EXTRACTION_USER_PROMPT,
        )
        from memory.extraction.validator import run_extraction
        from memory.profiles.registry import select_profile

        procs = process_names if process_names is not None else self.current_minute_processes
        titles = window_titles if window_titles is not None else self.current_minute_apps
        # Step 3: route to a domain profile (process_name preferred, title fallback).
        profile = select_profile(procs, titles)
        # Past user merges/renames steer the naming, so corrections stick.
        naming_hints = self.pipeline.naming_hints() if self.pipeline is not None else None
        system_prompt = build_system_prompt(profile, naming_hints=naming_hints)

        if len(imgs) == 0:
            return ExtractionResult(summary="", confidence=0.0), "empty", profile.name

        # Step 4: send a temporal spread of the whole gap, not just all frames.
        if self.gap_sampling and len(imgs) > self.gap_sample_count:
            from memory.sampling.gap_sampler import sample_gap_frames
            span_end = time.time()
            span_start = span_end - max(0, (len(imgs) - 1)) / max(self.fps, 1e-6)
            sample = sample_gap_frames(imgs, span_start, span_end, count=self.gap_sample_count)
            logger.info("Gap sample: %d/%d frames, timestamps=%s",
                        len(sample.frames), len(imgs), [round(t, 1) for t in sample.timestamps])
            imgs = sample.frames

        image_parts, frame_info = frames_as_image_parts(imgs)
        if not image_parts:
            return ExtractionResult(summary="Error preparing screen frames", confidence=0.0), "empty", profile.name

        base_content = [
            {"type": "text", "text": EXTRACTION_USER_PROMPT},
            *image_parts,
        ]

        async def generate(feedback):
            user_content = list(base_content)
            if feedback:
                user_content = [{"type": "text", "text": feedback}] + user_content
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            resp = await client.chat.completions.create(
                model=self.model_name_vlm, messages=messages,
                max_tokens=self.max_tokens if self.thinking else 2500,
                **thinking_request_kwargs(self.thinking),
            )
            return resp.choices[0].dict()["message"]["content"]

        tim = time.perf_counter()
        result, status = await run_extraction(generate)
        logger.info("Structured extraction (%s, profile=%s) using %d/%d images took %.3f s",
                    status, profile.name, frame_info["kept"], len(imgs),
                    time.perf_counter() - tim)
        return result, status, profile.name

    def _encode_buffer_to_mp4_base64(self,frames, fps=1.0):
        """Converts a list of numpy frames to a base64 encoded MP4 video."""
        if not frames:
            return None

        # 1. Resize for Speed (Target ~448p or 512p for speed)
        # Qwen3 likes multiples of 16. 448x448 is a sweet spot.
        # target_size = (448, 448)
        # resized_frames = [cv2.resize(f, target_size) for f in frames]

        height, width, layers = frames[0].shape

        # 2. Create Temp File
        # OpenCV VideoWriter usually needs a real file path
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_filename = f.name

        try:
            # 3. Write Frames to MP4
            # H.264 where the build has it, mp4v otherwise — see video_writer.
            out = open_mp4_writer(temp_filename, fps, (width, height))
            if out is None:
                logger.warning("Could not open a video writer for screen frames.")
                return None

            for frame in frames:
                # MSS captures BGRA/RGB, OpenCV expects BGR
                # Assuming your buffer is already RGB (from your capture code)
                # But VideoWriter expects BGR.
                numpy_image_rgb = np.array(frame)

                bgr_frame = cv2.cvtColor(numpy_image_rgb, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            out.release()

            # 4. Read back as Base64
            with open(temp_filename, "rb") as video_file:
                video_bytes = video_file.read()
                base64_video = base64.b64encode(video_bytes).decode('utf-8')

        finally:
            # Cleanup temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        return base64_video

    def _enqueue_batch(self, frames, timestamp, window_titles, process_names,
                       last_input_at=None):
        """Put a batch in the 1-slot mailbox, coalescing (dropping) any pending one."""
        payload = (frames, timestamp, list(window_titles), list(process_names),
                   last_input_at)
        with self._mailbox_lock:
            if self._mailbox is not None:
                self._dropped_batches += 1
                logger.warning("Worker busy — coalescing, dropped stale batch "
                               "(total dropped=%d)", self._dropped_batches)
            self._mailbox = payload
        self._mailbox_wake.set()

    def _batch_worker(self):
        """Single consumer: drains the mailbox sequentially (order guaranteed)."""
        while self.running:
            self._mailbox_wake.wait(timeout=1.0)
            with self._mailbox_lock:
                payload = self._mailbox
                self._mailbox = None
                self._mailbox_wake.clear()
            if payload is None:
                continue
            try:
                self._process_batch(*payload)
            except Exception as exc:
                logger.exception("batch processing failed: %s", exc)

    def _process_batch(self, imgs, timestamp, window_titles, process_names,
                       last_input_at=None):
        """Extract + (live) feed the memory pipeline for one batch."""
        logger.debug('Screen buffer %s processing started', timestamp)
        do_structured = self.structured_extraction or self.pipeline is not None

        result = None
        profile_name = None
        if do_structured:
            result, status, profile_name = asyncio.run(
                self._extract_structured(imgs, process_names, window_titles))
            description = result.summary
            record = result.model_dump()
            record.update({
                "timestamp": timestamp, "status": status,
                "selected_profile": profile_name,
                "process_names": list(process_names),
                "window_titles": list(window_titles),
            })
            try:
                write_jsonl("extractions", record)
            except Exception as exc:
                logger.debug("failed writing extraction: %s", exc)
        else:
            description = asyncio.run(self._describe_frames(imgs))

        # Recorded before ingest so the notification raised inside ingest can
        # already point at the footage it was raised from.
        clip_id = None
        if self.clip_store is not None and imgs:
            clip_id = self.clip_store.save(
                imgs, source="screen",
                label=(process_names[-1] if process_names else "screen"),
                timestamp=timestamp, capture_fps=self.fps, summary=description,
                extra={"window_titles": list(window_titles)[-3:]})

        if self.pipeline is not None and result is not None:
            # Live memory: sessions/events/knowledge + event-scoped dual store.
            try:
                ingested = self.pipeline.ingest({
                    "timestamp": timestamp,
                    "window_titles": list(window_titles),
                    "process_names": list(process_names),
                    "repr_frame": imgs[-1] if imgs else None,
                    "clip_id": clip_id,
                    "last_input_at": last_input_at,
                    "extraction": {**result.model_dump(), "selected_profile": profile_name},
                })
                if clip_id and self.clip_store is not None:
                    self.clip_store.annotate(
                        clip_id, event_id=ingested.current_event.event_id)
            except Exception as exc:
                logger.warning("pipeline.ingest failed: %s", exc)
        elif self.activity_logger is not None:
            # Legacy per-minute Qdrant logging (unchanged when LIVE_MEMORY off).
            self.activity_logger.log_activity(description, timestamp, 'screen', window_titles)

        if self.insight_callback is not None:
            try:
                self.insight_callback(description, timestamp, {"clip_id": clip_id})
            except Exception as exc:
                logger.warning("insight_callback failed: %s", exc)
        logger.debug('Screen buffer %s processing ended', timestamp)

    def _changed_fraction(self, img1, img2):
        """Share of pixels that differ between two frames, or None if incomparable.

        Also the presence signal for watching: a video repaints most of this every
        second, while an abandoned desktop's clock or graph repaints a fraction of
        a percent (sources.idle.PresenceGate).
        """
        if img1 is None or img2 is None:
            return None

        # Convert to grayscale for faster and more robust comparison
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # Compute the absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Count non-zero pixels (pixels that are different)
        return np.count_nonzero(diff) / diff.size

    def _are_images_similar(self, img1, img2, threshold=0.999):
        """
        Compares two images and returns True if their similarity is above the threshold.
        """
        changed = self._changed_fraction(img1, img2)
        if changed is None:
            return False
        similarity = 1.0 - changed
        logger.debug('Screen similarity: %.6f', similarity)
        return similarity > threshold

    def _track_active_window(self, monitor):
        """Sample the active window each capture iteration.

        Appends its title to current_minute_apps (feeds the describe path) and,
        when the active window changes, emits a structured Observation to
        data/debug/observations.jsonl (Step 1).
        """
        try:
            window = gw.getActiveWindow()
        except Exception as exc:
            logger.debug("getActiveWindow failed: %s", exc)
            return
        if not window or not window.title:
            return
        try:
            wx, wy = window.center
            in_x = monitor["left"] <= wx < (monitor["left"] + monitor["width"])
            in_y = monitor["top"] <= wy < (monitor["top"] + monitor["height"])
            in_bounds = in_x and in_y
        except Exception:
            in_bounds = True  # can't tell — don't drop the title
        if not in_bounds:
            if self._strict_monitor_bounds:
                return
            logger.debug("active window '%s' center outside captured monitor "
                         "bounds — recording anyway", window.title[:40])

        if window.title not in self.current_minute_apps:
            self.current_minute_apps.append(window.title)

        process_name, application = _process_for_hwnd(getattr(window, "_hWnd", None))
        if process_name and process_name not in self.current_minute_processes:
            self.current_minute_processes.append(process_name)
        key = (window.title, process_name)
        if key != self._last_observed_key:
            self._last_observed_key = key
            obs = Observation(
                timestamp=time.time(),
                process_name=process_name,
                application=application,
                window_title=window.title,
                screen_id=self.monitor_index,
            )
            try:
                write_jsonl("observations", obs)
            except Exception as exc:
                logger.debug("failed writing observation: %s", exc)

    def _capture_frames(self):
        with mss() as sct:
            try:
                monitor = sct.monitors[self.monitor_index]
            except IndexError:
                self.last_error = f"monitor index {self.monitor_index} not found"
                logger.error(self.last_error)
                self.running = False
                return
            self.healthy = True
            profile_version = self._profile_version
            next_batch_at = time.monotonic() + self.window_size
            last_frame = None
            # The previous grab, kept only to measure how much repaints per
            # second — separate from last_frame, which is the last frame kept.
            previous_grab = None

            while self.running:
                if profile_version != self._profile_version:
                    profile_version = self._profile_version
                    next_batch_at = time.monotonic() + self.window_size
                    last_frame = None
                    previous_grab = None
                # Paused: hold the thread but capture/process nothing. Reset the
                # per-minute state so we don't emit a stale batch on resume.
                if self.paused:
                    if self.current_minute_apps or self.current_minute_processes:
                        with self.lock:
                            self.frame_buffer.clear()
                        self.current_minute_apps = list()
                        self.current_minute_processes = list()
                        last_frame = None
                        previous_grab = None
                        self._minute_changes = []
                    self._profile_changed.wait(0.3)
                    self._profile_changed.clear()
                    continue

                # Poll the active window every iteration so window switches during
                # the minute are all captured (not just the one focused at start).
                self._track_active_window(monitor)

                # Capture the screen
                screenshot = sct.grab(monitor)

                # Convert to numpy array
                img = np.array(screenshot)

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

                # Resize if target resolution is specified
                if self.target_resolution:
                    img = cv2.resize(img, self.target_resolution, interpolation=cv2.INTER_AREA)

                # How much of the screen repainted since the *previous grab* —
                # measured against every frame, not just the last kept one, so a
                # slowly drifting picture can't accumulate into false motion.
                # This is what tells watching apart from an abandoned desk.
                change = self._changed_fraction(previous_grab, img)
                if change is not None:
                    self._minute_changes.append(change)
                previous_grab = img

                # Only add frame if it's different enough from the last one
                if not self._are_images_similar(last_frame, img):
                    with self.lock:
                        self.frame_buffer.append(img)
                    last_frame = img
                    logger.debug('buffer = %d (new frame added)', len(self.frame_buffer))

                if time.monotonic() >= next_batch_at:
                    with self.lock:
                        # Always snapshot + keep the last 2 frames as overlap, so a
                        # heartbeat still has a frame even on a static screen.
                        frames = list(self.frame_buffer)
                        overlap_frames = frames[-2:]
                        self.frame_buffer.clear()
                        self.frame_buffer.extend(overlap_frames)

                    presence = self.presence_gate.evaluate(self._minute_changes)
                    self._minute_changes = []
                    logger.debug("Presence: %s (idle %.0fs, screen repaint %s)",
                                 presence.reason, presence.idle_seconds or 0.0,
                                 "n/a" if presence.change_level is None
                                 else f"{presence.change_level * 100:.2f}%")
                    if not presence.present:
                        # Away: drop the minute. The per-minute window state goes
                        # too, so the first minute back opens a fresh event rather
                        # than extending the one the user walked away from.
                        self._away_minutes += 1
                        if self._away_minutes == 1:
                            logger.info(
                                "No keyboard/mouse input for %.0fs (> %.0fs) and "
                                "the screen is not playing anything — capture "
                                "paused until the user is back.",
                                presence.idle_seconds or 0.0,
                                self.presence_gate.input.timeout_seconds)
                        self.current_minute_apps = list()
                        self.current_minute_processes = list()
                        self._idle_minutes = 0
                        last_frame = None
                        previous_grab = None
                        with self.lock:
                            # Drop the overlap frames as well — they are from
                            # before the user left and would leak into the first
                            # batch after they return.
                            self.frame_buffer.clear()
                        next_batch_at = time.monotonic() + self.window_size
                        self._profile_changed.wait(1.0 / self.fps)
                        self._profile_changed.clear()
                        continue
                    if self._away_minutes:
                        logger.info("User back after %d away minute(s) — resuming capture.",
                                    self._away_minutes)
                        self._away_minutes = 0
                    if presence.watching and self.presence_gate.playback_minutes == 1:
                        logger.info(
                            "No input for %.0fs but %.1f%% of the screen is "
                            "repainting each second — counting this as watching.",
                            presence.idle_seconds or 0.0,
                            (presence.change_level or 0.0) * 100.0)

                    enough_activity = len(frames) > 2
                    active_window = bool(self.current_minute_apps)
                    if enough_activity:
                        self._idle_minutes = 0
                        heartbeat = False
                    else:
                        self._idle_minutes += 1
                        heartbeat = (active_window and self.heartbeat_minutes > 0
                                     and self._idle_minutes >= self.heartbeat_minutes)

                    can_process = self._need_process or self.recorder is not None
                    if can_process and (enough_activity or heartbeat) and len(frames) >= 1:
                        frames_to_process = frames
                        if heartbeat:
                            logger.info("Heartbeat capture after %d idle minute(s).",
                                        self._idle_minutes)
                            self._idle_minutes = 0
                    else:
                        frames_to_process = []
                    if frames_to_process:
                        batch_ts = time.time()
                        # Snapshot this minute's window/process context.
                        titles = list(self.current_minute_apps)
                        procs = list(self.current_minute_processes)
                        # Step 0: record this batch to disk for offline replay.
                        if self.recorder is not None:
                            self.recorder.record_batch(frames_to_process, batch_ts, titles, procs)
                        # Live processing via the coalescing mailbox (worker drains it).
                        if self._need_process:
                            self._enqueue_batch(frames_to_process, batch_ts, titles,
                                                procs, presence.active_until)
                        # Reset for the next minute, keeping the last title/process
                        # as carryover (the worker uses the snapshot above).
                        self.current_minute_apps = self.current_minute_apps[-1:]
                        self.current_minute_processes = self.current_minute_processes[-1:]
                    next_batch_at = time.monotonic() + self.window_size
                # Handle dynamic framerates (adjust as needed)
                self._profile_changed.wait(1.0 / self.fps)
                self._profile_changed.clear()


    # todo, not sure for now  how i want to implement it, postponed
    def new_activity(self):
        if self.activity_logger is not None:
            self.activity_logger.reset()
            # self.description_history.clear()

    def pause(self):
        """Stop capturing/processing without tearing down the thread."""
        self.paused = True
        logger.info("Screen capture paused.")

    def resume(self):
        self.paused = False
        logger.info("Screen capture resumed.")

    def update_capture_profile(self, sample_fps, inference_interval_seconds,
                               thinking=None):
        """Apply a new profile and discard the current/pending capture window."""
        fps, interval = validate_capture_profile(
            sample_fps, inference_interval_seconds
        )
        with self.lock:
            self.fps = fps
            self.window_size = interval
            if thinking is not None:
                self.thinking = bool(thinking)
            self.frame_buffer = deque(
                maxlen=expected_frame_count(fps, interval)
            )
            self.current_minute_apps = []
            self.current_minute_processes = []
            self._minute_changes = []
            self._profile_version += 1
        with self._mailbox_lock:
            self._mailbox = None
            self._mailbox_wake.clear()
        if self.pipeline is not None:
            self.pipeline.expected_seconds = interval
        self._idle_minutes = 0
        self._profile_changed.set()
        logger.info(
            "Screen capture profile changed to %.3g fps / %ds; "
            "frame buffer reset.",
            fps,
            interval,
        )

    def cleanup(self):
        self.running = False
        self._profile_changed.set()
        self._mailbox_wake.set()  # unblock the worker so it can exit
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=5)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=10)
        if self.describe_thread is not None:
            self.describe_thread.join(timeout=5)
        # Flush the last open event so the final partial minute persists.
        if self.pipeline is not None:
            try:
                self.pipeline.finalize()
            except Exception as exc:
                logger.warning("pipeline.finalize failed: %s", exc)

    def frames(self):
        with self.lock:
            return list(self.frame_buffer)

    def status(self):
        alive = self.capture_thread is not None and self.capture_thread.is_alive()
        idle_state = self.presence_gate.input.state()
        return {
            "input_idle_seconds": (None if idle_state.idle_seconds is None
                                   else round(idle_state.idle_seconds, 1)),
            "input_idle_timeout_seconds": self.presence_gate.input.timeout_seconds,
            "user_away": self._away_minutes > 0,
            "away_minutes": self._away_minutes,
            "watching_minutes": self.presence_gate.playback_minutes,
            "playback_change_fraction": self.presence_gate.change_fraction,
            "configured": True,
            "healthy": self.healthy and alive and not self.paused,
            "running": self.running,
            "paused": self.paused,
            "frames": len(self.frame_buffer),
            "monitor_index": self.monitor_index,
            "sample_fps": self.fps,
            "inference_interval_seconds": self.window_size,
            "thinking": self.thinking,
            "expected_frames": expected_frame_count(
                self.fps, self.window_size
            ),
            "error": self.last_error,
        }

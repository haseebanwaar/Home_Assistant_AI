"""RTSP frame reader.

Two settings here decide whether the rest of the pipeline sees a picture or
garbage, so they are applied before the first `cv2.VideoCapture` is built:

`rtsp_transport;tcp` — RTSP defaults to UDP, where a lost packet costs the
decoder a reference frame.

`CAMERA_STREAM` — main (default) or sub. Measured on both cameras, the HEVC main
stream (`subtype=0`) sustains only 6-7 fps of software decode against a ~15 fps
source and emits a continuous run of `Could not find ref with POC` errors; the
visible result is the flat grey and macroblock frames the VLM kept narrating
("the view transitions from a heavily pixelated, gray screen…"), with a peak
frame-to-frame delta of 52-118 grey levels that no motion gate can distinguish
from a real event. The substream (`subtype=1`) decodes at full rate with ZERO
decoder errors and a peak delta of 3.0 — but it is not the default, because
these are HD cameras and resolution is what they are for. Corruption that gets
through is caught after extraction instead; see `sources.camera_validation`.

`flat_frame_pct` in `status()` is how you tell a decode problem from a quiet
scene without reading VLM summaries.
"""
import os
import re
import time
import logging
from collections import deque
from threading import Event, Thread, Lock

import cv2
from PIL import Image

logger = logging.getLogger("home_assistant")

# Set once at import: OpenCV reads this env var when a VideoCapture is created,
# so setting it per-capture would race between camera threads.
_RTSP_TRANSPORT = os.getenv("CAMERA_RTSP_TRANSPORT", "tcp").strip()
if _RTSP_TRANSPORT and "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{_RTSP_TRANSPORT}"

# Vendor patterns for "same camera, lighter stream". Anything that matches
# neither is passed through untouched rather than guessed at.
_SUBSTREAM_REWRITES = (
    (re.compile(r"(?i)(\bsubtype=)0\b"), r"\g<1>1"),              # Dahua/Amcrest
    (re.compile(r"(?i)(/Streaming/Channels/\d*)1\b"), r"\g<1>2"),  # Hikvision
)


# Grey-level stddev below this reads as a frame with no picture in it. Measured:
# clean substream p05 = 46.2 (cam .7) / 46.8 (cam .17); broken main-stream decode
# p05 = 2.0 / 0.7. Only used for reporting, never to discard a frame.
FLAT_FRAME_STD = float(os.getenv("CAMERA_FLAT_FRAME_STD", "8.0"))


def _redact(url):
    """RTSP URLs carry credentials; keep them out of the log."""
    return re.sub(r"//[^/@]*@", "//***@", str(url or ""))


def prefer_substream(url):
    """Rewrite an RTSP URL to the camera's substream (see module docstring).

    Off by default: the substream decodes cleanly but costs resolution, and these
    are HD cameras chosen for that resolution — dropping it to dodge a decode
    problem would quietly degrade every extraction. `CAMERA_STREAM=sub` opts in
    for a camera where picture quality matters less than a clean decode.
    """
    if os.getenv("CAMERA_STREAM", "main").strip().lower() != "sub":
        return url
    for pattern, replacement in _SUBSTREAM_REWRITES:
        new_url, count = pattern.subn(replacement, url or "", count=1)
        if count:
            return new_url
    return url


class RealtimeCameraStream:
    def __init__(self, video_source, window_size=10, fps=1.0,
                 reconnect_initial_delay=0.25, reconnect_max_delay=5.0):
        """
        Args:
            video_source: RTSP URL or video path
            window_size: Number of seconds to keep in memory
            fps: Frames per second to process
        """
        self.video_source = prefer_substream(video_source)
        if self.video_source != video_source:
            logger.info("Using camera substream: %s", _redact(self.video_source))
        self.window_size = window_size
        self.fps = fps
        # Flat-frame telemetry. A healthy substream frame measures a grey-level
        # stddev around 46; a decoder frame that lost its reference collapses to
        # ~1-2. Counting them (rather than dropping them) surfaces a degrading
        # stream in /cameras without risking a genuinely dark night scene being
        # thrown away on an uncalibrated threshold.
        self.flat_frames = 0
        self.total_frames = 0
        self.last_frame_std = None
        self.frame_buffer = deque(maxlen=window_size)
        self.lock = Lock()
        self._profile_version = 0
        self.running = True
        self.healthy = False
        self.last_error = None
        self.last_frame_at = None
        self.reconnect_count = 0
        self._reconnect_initial_delay = max(float(reconnect_initial_delay), 0.0)
        self._reconnect_max_delay = max(
            float(reconnect_max_delay), self._reconnect_initial_delay)
        self._stop = Event()

        # Start frame capture thread
        self.capture_thread = Thread(
            target=self._capture_frames, name="rtsp-capture")
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def _capture_frames(self):
        retry_delay = self._reconnect_initial_delay
        connected_once = False
        while self.running and not self._stop.is_set():
            video = None
            try:
                video = cv2.VideoCapture(self.video_source)
                if not video.isOpened():
                    self._mark_disconnected("camera source could not be opened")
                else:
                    connection_had_frame = False
                    next_sample_at = None
                    profile_version = self._profile_version
                    while self.running and not self._stop.is_set():
                        ret, frame = video.read()
                        if not ret:
                            self._mark_disconnected(
                                "camera stream stopped returning frames")
                            break

                        # Do not report a connection as healthy until it has
                        # actually delivered a frame.
                        if not connection_had_frame:
                            connection_had_frame = True
                            if connected_once:
                                self.reconnect_count += 1
                                logger.info("Camera stream reconnected.")
                            connected_once = True
                            retry_delay = self._reconnect_initial_delay
                            self.last_error = None
                            self.healthy = True

                        self._record_quality(frame)
                        self.last_frame_at = time.time()

                        # Always drain/decode the RTSP stream. Inter-frame codecs
                        # such as HEVC need every reference frame even though the
                        # VLM only needs a sparse sample. Sleeping here used to
                        # leave ~14 of every 15 camera frames unread, eventually
                        # producing grey macroblocks and missing-POC errors.
                        now = time.monotonic()
                        if profile_version != self._profile_version:
                            profile_version = self._profile_version
                            next_sample_at = None
                        if next_sample_at is None or now >= next_sample_at:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(frame_rgb)
                            with self.lock:
                                self.frame_buffer.append(pil_image)
                            sample_interval = 1.0 / max(self.fps, 0.01)
                            next_sample_at = now + sample_interval
            except Exception as exc:
                self._mark_disconnected(f"camera stream error: {exc}")
                logger.exception("Camera stream capture failed")
            finally:
                if video is not None:
                    video.release()
                self.healthy = False

            if self.running and not self._stop.is_set():
                if self._stop.wait(retry_delay):
                    break
                retry_delay = min(
                    max(retry_delay * 2, self._reconnect_initial_delay),
                    self._reconnect_max_delay,
                )

    def _record_quality(self, frame):
        """Track how many frames come back visually flat (see __init__).

        Measured on a 1-in-8 subsample so this stays negligible at full frame
        rate across several cameras.
        """
        try:
            sub = frame[::8, ::8]
            std = float(sub.std())
        except Exception:
            return
        self.last_frame_std = round(std, 2)
        self.total_frames += 1
        if std < FLAT_FRAME_STD:
            self.flat_frames += 1

    def _mark_disconnected(self, error):
        was_healthy = self.healthy
        self.healthy = False
        self.last_error = error
        # Never let downstream processing mistake buffered pre-drop frames for
        # a live view. A recovered stream will refill this within a few samples.
        with self.lock:
            self.frame_buffer.clear()
        log = logger.warning if was_healthy else logger.info
        log("%s; reconnecting.", error)

    def cleanup(self):
        self.running = False
        self._stop.set()
        self.capture_thread.join(timeout=5)

    def update_sampling(self, fps, buffer_frames):
        """Apply a new sampling rate and start a fresh bounded frame window."""
        fps = max(float(fps), 0.01)
        buffer_frames = max(int(buffer_frames), 1)
        with self.lock:
            self.fps = fps
            self.frame_buffer = deque(maxlen=buffer_frames)
            self._profile_version += 1

    def frames(self):
        with self.lock:
            return list(self.frame_buffer)

    def status(self):
        return {
            "configured": True,
            "healthy": self.healthy and self.capture_thread.is_alive(),
            "running": self.running,
            "frames": len(self.frame_buffer),
            "error": self.last_error,
            "last_frame_at": self.last_frame_at,
            "reconnects": self.reconnect_count,
            "flat_frame_pct": (round(100.0 * self.flat_frames / self.total_frames, 1)
                               if self.total_frames else None),
            "last_frame_std": self.last_frame_std,
            "sample_fps": self.fps,
            "buffer_capacity": self.frame_buffer.maxlen,
        }

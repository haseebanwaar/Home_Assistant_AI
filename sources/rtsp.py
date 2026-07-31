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
scene without reading VLM summaries. It is measured over sampled frames (see
`_record_quality`), so it reports the quality of what the VLM was shown.

Draining and sampling are deliberately separated: every frame is grabbed so the
decoder keeps its references, but only frames at `fps` are retrieved into
pixels. Reading (grab+retrieve) every frame instead cost a full CPU core per
pair of HD cameras to produce frames that were immediately discarded.
"""
import os
import re
import time
import logging
from collections import deque
from threading import Event, Thread, Lock

import cv2
from PIL import Image
from line_profiler import profile


logger = logging.getLogger("home_assistant")

# Set once at import: OpenCV reads this env var when a VideoCapture is created,
# so setting it per-capture would race between camera threads.
_RTSP_TRANSPORT = os.getenv("CAMERA_RTSP_TRANSPORT", "tcp").strip()
if _RTSP_TRANSPORT and "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{_RTSP_TRANSPORT}"

# FFmpeg frame-threads per decoder. Left to FFmpeg, HEVC picks one thread per
# core: on a 12-core box two cameras opened 24 decoder threads that each burned
# 1-2% of a core purely in scheduling overhead. Two threads per camera is enough
# to stay ahead of a ~15 fps HD stream. 0 restores FFmpeg's own choice.
# Raise this (and watch `flat_frame_pct`) if a stream starts falling behind.
_DECODER_THREADS = int(os.getenv("CAMERA_DECODER_THREADS", "2"))

# Decoder acceleration is an open-only OpenCV property, like N_THREADS. On the
# tested Windows/FFmpeg build, ANY selects a working HEVC hardware path and cuts
# main-stream decode CPU without sacrificing resolution. "none" forces software.
_HW_ACCELERATION_NAME = os.getenv(
    "CAMERA_HW_ACCELERATION", "any").strip().lower()
_HW_ACCELERATIONS = {
    "none": getattr(cv2, "VIDEO_ACCELERATION_NONE", 0),
    "any": getattr(cv2, "VIDEO_ACCELERATION_ANY", 1),
    "d3d11": getattr(cv2, "VIDEO_ACCELERATION_D3D11", 2),
    "vaapi": getattr(cv2, "VIDEO_ACCELERATION_VAAPI", 3),
    "mfx": getattr(cv2, "VIDEO_ACCELERATION_MFX", 4),
}

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


def _open_capture(video_source):
    """Open FFmpeg with all open-only decoder options applied correctly.

    CAP_PROP_FOURCC is intentionally not set: an RTSP client's FourCC cannot
    transcode the camera's HEVC payload. Measured on this build -- `set()`
    returns False for all of mp4v/MJPG/H264/avc1/HEVC/hvc1/I420/YUY2/X264 and
    the property keeps reading back `hevc`, with no change in decode cost. On the
    FFmpeg backend the FourCC only *reports* the stream; it is a device request
    (MJPG vs YUY2) on webcam backends, which is not what this is.

    The levers that do work, per-frame CPU at 2560x1440, 300 frames:

        accel=none    grab 9.64ms   retrieve  9.90ms
        accel=any     grab 0.89ms   retrieve 16.56ms

    Acceleration cuts demux+decode ~11x but pays for a GPU->CPU readback on
    retrieve, so it wins precisely because this loop grabs every frame and
    retrieves ~1 per second: ~39ms/s of CPU versus ~251ms/s for software.
    Thread count made no measurable difference at 1/2/4/auto. Select
    CAMERA_STREAM=sub (~6x cheaper again) when resolution matters less.
    """
    params = []
    if _DECODER_THREADS > 0 and hasattr(cv2, "CAP_PROP_N_THREADS"):
        params.extend((cv2.CAP_PROP_N_THREADS, _DECODER_THREADS))
    acceleration = _HW_ACCELERATIONS.get(_HW_ACCELERATION_NAME)
    if acceleration is None:
        logger.warning(
            "Unknown CAMERA_HW_ACCELERATION=%r; using any.",
            _HW_ACCELERATION_NAME,
        )
        acceleration = _HW_ACCELERATIONS["any"]
    if (
        _HW_ACCELERATION_NAME != "none"
        and hasattr(cv2, "CAP_PROP_HW_ACCELERATION")
    ):
        params.extend((cv2.CAP_PROP_HW_ACCELERATION, acceleration))

    if params:
        video = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG, params)
        if video.isOpened():
            return video
        video.release()
        logger.warning(
            "Camera rejected FFmpeg decoder options; retrying with defaults.")
    return cv2.VideoCapture(video_source)


def _fourcc_name(value):
    number = int(value or 0)
    return "".join(
        chr((number >> (8 * index)) & 0xFF) for index in range(4)
    ).rstrip("\x00").lower()


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
        self.codec = None
        self.frame_width = None
        self.frame_height = None
        self.source_fps = None
        self.decoder_threads = None
        self.hw_acceleration = None
        self._reconnect_initial_delay = max(float(reconnect_initial_delay), 0.0)
        self._reconnect_max_delay = max(
            float(reconnect_max_delay), self._reconnect_initial_delay)
        self._stop = Event()

        # Start frame capture thread
        self.capture_thread = Thread(
            target=self._capture_frames, name="rtsp-capture")
        self.capture_thread.daemon = True
        self.capture_thread.start()

    @profile
    def _capture_frames(self):
        retry_delay = self._reconnect_initial_delay
        connected_once = False
        while self.running and not self._stop.is_set():
            video = None
            try:
                video = _open_capture(self.video_source)
                if not video.isOpened():
                    self._mark_disconnected("camera source could not be opened")
                else:
                    self._record_decoder(video)
                    connection_had_frame = False
                    next_sample_at = None
                    profile_version = self._profile_version
                    while self.running and not self._stop.is_set():
                        # Always drain the RTSP stream. Inter-frame codecs such
                        # as HEVC need every reference frame even though the VLM
                        # only needs a sparse sample. Sleeping here used to leave
                        # ~14 of every 15 camera frames unread, eventually
                        # producing grey macroblocks and missing-POC errors.
                        if not video.grab():
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

                        # Liveness is about the stream, not the sample, so it is
                        # stamped on every grabbed frame.
                        self.last_frame_at = time.time()

                        now = time.monotonic()
                        if profile_version != self._profile_version:
                            profile_version = self._profile_version
                            next_sample_at = None
                        if next_sample_at is not None and now < next_sample_at:
                            continue

                        # Only the frames we keep are converted to pixels.
                        # `grab` demuxes and decodes; `retrieve` is the YUV->BGR
                        # copy, and paying it 15x per second per camera to throw
                        # 14 of those frames away is what pegged a whole core.
                        # The sampling clock advances either way: a frame that
                        # will not decode must not turn this back into a
                        # full-rate retrieve loop.
                        next_sample_at = now + 1.0 / max(self.fps, 0.01)
                        ok, frame = video.retrieve()
                        if not ok:
                            continue

                        self._record_quality(frame)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        with self.lock:
                            self.frame_buffer.append(pil_image)
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

        Called only on sampled frames — the ones the VLM will actually see —
        because those are the ones whose quality matters, and because the
        unsampled frames are never decoded to pixels to measure. Measured on a
        1-in-8 subsample so it stays negligible across several cameras.
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

    def _record_decoder(self, video):
        """Expose the negotiated stream/decoder so CPU choices are observable."""
        try:
            self.codec = _fourcc_name(video.get(cv2.CAP_PROP_FOURCC)) or None
            self.frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            self.frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            self.source_fps = round(
                float(video.get(cv2.CAP_PROP_FPS) or 0), 1) or None
            if hasattr(cv2, "CAP_PROP_N_THREADS"):
                self.decoder_threads = int(
                    video.get(cv2.CAP_PROP_N_THREADS) or 0) or None
            if hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
                self.hw_acceleration = int(
                    video.get(cv2.CAP_PROP_HW_ACCELERATION) or 0)
            logger.info(
                "Camera decoder: codec=%s size=%sx%s source_fps=%s "
                "threads=%s hw=%s",
                self.codec, self.frame_width, self.frame_height,
                self.source_fps, self.decoder_threads, self.hw_acceleration,
            )
        except Exception:
            logger.debug("Could not read camera decoder metadata", exc_info=True)

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
            "codec": self.codec,
            "resolution": (
                f"{self.frame_width}x{self.frame_height}"
                if self.frame_width and self.frame_height else None
            ),
            "source_fps": self.source_fps,
            "decoder_threads": self.decoder_threads,
            "hw_acceleration": self.hw_acceleration,
        }

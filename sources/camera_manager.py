"""Camera manager — discover live cameras and run a worker per camera.

Discovery reuses utils.camera_discovery (ONVIF WS-Discovery + credential probe
from CAMERA_CREDENTIALS) to find cameras that are actually live and hand back an
RTSP URL. For every such camera we ensure a dedicated Room and start a
CameraCaptureWorker. Any URLs in CAMERA_RTSP_URL / CAMERA_RTSP_URLS are added as
explicit cameras too, so a known stream works without discovery.
"""
from __future__ import annotations

import logging
import os
import re

from sources.camera_worker import CameraCaptureWorker

logger = logging.getLogger("home_assistant")


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-") or "camera"


def _discover_live_cameras():
    """Return [{camera_id, name, rtsp_url}] for live, addressable cameras."""
    cameras = []
    seen = set()

    # Explicit URLs first (CAMERA_RTSP_URLS="name|rtsp://..,name2|rtsp://.." or a
    # bare CAMERA_RTSP_URL) — these never depend on discovery succeeding.
    explicit = os.getenv("CAMERA_RTSP_URLS", "")
    single = os.getenv("CAMERA_RTSP_URL", "")
    entries = [e for e in explicit.split(",") if e.strip()]
    if single.strip():
        entries.append(single.strip())
    for entry in entries:
        if "|" in entry:
            name, url = entry.split("|", 1)
        else:
            name, url = "", entry
        url = url.strip()
        if not url or url in seen:
            continue
        seen.add(url)
        name = name.strip() or _host_of(url) or "Camera"
        cameras.append({"camera_id": f"camera:{_slug(name)}", "name": name,
                        "rtsp_url": url})

    # ONVIF discovery for everything else on the network.
    try:
        from utils.camera_discovery import discover_onvif_and_identify
        devices = discover_onvif_and_identify()
    except Exception as exc:
        logger.warning("Camera discovery failed (%s) — using explicit URLs only.", exc)
        devices = []

    for dev in devices or []:
        rtsp = dev.get("rtsp_url")
        if not rtsp or rtsp in seen:
            continue
        seen.add(rtsp)
        info = dev.get("onvif_info") or {}
        model = info.get("model") or info.get("manufacturer")
        host = dev.get("host")
        name = model or host or "Camera"
        cam_id = f"camera:{_slug(host or name)}"
        cameras.append({"camera_id": cam_id, "name": name, "rtsp_url": rtsp})

    return cameras


def _host_of(url):
    m = re.search(r"@([^:/]+)", url) or re.search(r"//([^:/@]+)", url)
    return m.group(1) if m else None


class CameraManager:
    def __init__(self, model_name_vlm, neo4j_store=None, activity_logger=None,
                 window_seconds=60, fps=1.0, notification_sink=None,
                 insight_callback=None, clip_store=None):
        self.model_name_vlm = model_name_vlm
        self.neo4j = neo4j_store
        self.activity_logger = activity_logger
        self.window_seconds = window_seconds
        self.fps = fps
        self.notification_sink = notification_sink
        self.insight_callback = insight_callback
        self.clip_store = clip_store
        self.workers = {}  # camera_id -> CameraCaptureWorker

    def discover_and_start(self):
        """Blocking: discover live cameras and start a worker for each."""
        cameras = _discover_live_cameras()
        if not cameras:
            logger.info("No live cameras discovered/configured.")
            return []
        for cam in cameras:
            self._start_one(cam["camera_id"], cam["name"], cam["rtsp_url"])
        logger.info("Camera manager running %d camera(s): %s",
                    len(self.workers), ", ".join(self.workers))
        return list(self.workers)

    def _start_one(self, camera_id, name, rtsp_url):
        if camera_id in self.workers:
            return
        # Pre-create the shared Cameras room so the first event routes there
        # (routing only sees rooms that already exist in the graph). Every camera
        # shares it; the camera's name is the tag on each event.
        if self.neo4j is not None:
            try:
                self.neo4j.ensure_source_room("camera")
            except Exception as exc:
                logger.warning("ensure_source_room(camera) failed: %s", exc)
        try:
            self.workers[camera_id] = CameraCaptureWorker(
                camera_id=camera_id, name=name, rtsp_url=rtsp_url,
                model_name_vlm=self.model_name_vlm, neo4j_store=self.neo4j,
                activity_logger=self.activity_logger,
                window_seconds=self.window_seconds, fps=self.fps,
                notification_sink=self.notification_sink,
                insight_callback=self.insight_callback,
                clip_store=self.clip_store)
            logger.info("Started camera worker %s (%s).", camera_id, name)
        except Exception as exc:
            logger.warning("failed to start camera %s: %s", camera_id, exc)

    def pause(self, camera_id):
        w = self.workers.get(camera_id)
        if w is None:
            return False
        w.pause()
        return True

    def resume(self, camera_id):
        w = self.workers.get(camera_id)
        if w is None:
            return False
        w.resume()
        return True

    def status_all(self):
        return [w.status() for w in self.workers.values()]

    def health_all(self):
        return [w.health() for w in self.workers.values()]

    def cleanup_all(self):
        for w in self.workers.values():
            try:
                w.cleanup()
            except Exception:
                logger.warning("camera %s cleanup failed", w.camera_id, exc_info=True)

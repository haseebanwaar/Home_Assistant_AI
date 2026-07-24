import time
import logging
from collections import deque
from threading import Thread, Lock

import cv2
from PIL import Image

logger = logging.getLogger("home_assistant")


class RealtimeCameraStream:
    def __init__(self, video_source, window_size=10, fps=1.0):
        """
        Args:
            video_source: RTSP URL or video path
            window_size: Number of seconds to keep in memory
            fps: Frames per second to process
        """
        self.video_source = video_source
        self.window_size = window_size
        self.fps = fps
        self.frame_buffer = deque(maxlen=window_size)
        self.lock = Lock()
        self.running = True
        self.healthy = False
        self.last_error = None

        # Start frame capture thread
        self.capture_thread = Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def _capture_frames(self):
        video = cv2.VideoCapture(self.video_source)

        if not video.isOpened():
            self.last_error = "camera source could not be opened"
            self.running = False
            logger.error(self.last_error)
            video.release()
            return

        self.healthy = True

        while self.running:
            ret, frame = video.read()
            if not ret:
                self.last_error = "camera stream stopped returning frames"
                self.healthy = False
                logger.warning(self.last_error)
                break

            # Convert frame to PIL Image for VLM compatibility
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            with self.lock:
                self.frame_buffer.append(pil_image)

            # Sample roughly self.fps frames per second.
            time.sleep(1.0 / self.fps)

        video.release()
        self.healthy = False

    def cleanup(self):
        self.running = False
        self.capture_thread.join(timeout=5)

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
        }


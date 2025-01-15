import time
from collections import deque
from threading import Thread, Lock

import cv2
from PIL import Image


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

        # Start frame capture thread
        self.capture_thread = Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def _capture_frames(self):
        video = cv2.VideoCapture(self.video_source)
        frame_count = 0

        while self.running:
            ret, frame = video.read()
            if not ret:
                break

            if frame_count % (24 // self.fps) == 0:  # Assuming 30fps video
                # Convert frame to PIL Image for VLM compatibility
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                timestamp = time.time()
                with self.lock:
                    # self.frame_buffer.append({
                    #     'image': pil_image,
                    #     'timestamp': timestamp
                    # })
                    self.frame_buffer.append(pil_image)
                time.sleep(0.99)

            frame_count += 1

        video.release()

    def cleanup(self):
        self.running = False
        self.capture_thread.join()


import base64
import math
import os
import time
import asyncio
from collections import deque
from threading import Thread, Lock
import cv2
from PIL import Image
from mss import mss
import numpy as np
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from providers.local_openAI import client, get_model_name_vlm
from utils.qwen_preprocess import encode_video
import cv2
import tempfile
import pygetwindow as gw

class RealtimeScreenCapture:
    def __init__(self, video_source,model_name_vlm, window_size=60, fps=1.0, monitor_index=1, target_resolution=None,
                 activity_logger=None):
        """
        Args:
            video_source: screen (not used)
            window_size: Number of seconds to keep in memory
            fps: Frames per second to process
            monitor_index: the index of the monitor to capture (default 1 for primary)
            target_resolution: a tuple of (width, height) for resizing, or None to keep original resolution
            activity_logger: an instance of ActivityLogger to log each minute of activity
        """
        self.video_source = video_source
        self.window_size = window_size
        self.fps = fps
        self.frame_buffer = deque(maxlen=window_size)
        self.lock = Lock()
        self.running = True
        self.monitor_index = monitor_index
        self.model_name_vlm = model_name_vlm
        self.target_resolution = target_resolution
        self.activity_logger = activity_logger
        self.current_minute_apps = list()
        self.describe_thread = None # Thread for describing frames
        self.describe_thread_lock = Lock()  # Lock for the thread
        # self.description_history = deque(maxlen=1)  #store last 3 description for context
        # Start frame capture thread
        self.capture_thread = Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    async def _describe_frames(self, imgs):
        """
        Describe the current minute of activity, using VLM
        """
        if len(imgs) == 0:
            return ''
        video_b64 = self._encode_buffer_to_mp4_base64(imgs,fps=self.fps)
        if not video_b64: return "Error encoding video"

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
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
                    }
                ]
            }
        ]
        tim = time.perf_counter()
        response = await client.chat.completions.create(model=self.model_name_vlm, messages=messages,max_tokens=2500)

        answer = response.choices[0].dict()['message']['content']
        print(answer)
        print(f"screen processing of {len(imgs)} frames took: {time.perf_counter()-tim}")


        return answer

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
            # 'mp4v' is widely supported. 'avc1' (H.264) is better if available.
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))

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

    def _describe_frames_threaded(self, imgs, timestamp):
        """
          This method is now threaded.
        """
        print(f'buffer = {timestamp} running')
        # try:
        description = asyncio.run(self._describe_frames(imgs))
        self.activity_logger.log_activity(description, timestamp,'screen',self.current_minute_apps)
        # with self.lock:
        #     self.description_history.append(description)
        if self.current_minute_apps:
            self.current_minute_apps = [self.current_minute_apps[-1]]
        else:
            self.current_minute_apps = []
        # except Exception as e:
        #     print(f"An error occurred in _describe_frames_threaded: {e}")
        print(f'buffer = {timestamp} ended')

    def _are_images_similar(self, img1, img2, threshold=0.999):
        """
        Compares two images and returns True if their similarity is above the threshold.
        """
        if img1 is None or img2 is None:
            return False

        # Convert to grayscale for faster and more robust comparison
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # Compute the absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Count non-zero pixels (pixels that are different)
        non_zero_count = np.count_nonzero(diff)
        total_pixels = diff.size
        similarity = (total_pixels - non_zero_count) / total_pixels
        print(similarity)
        return similarity > threshold

    def _capture_frames(self):
        with mss() as sct:
            try:
                monitor = sct.monitors[self.monitor_index]
            except IndexError:
                print(f"Error: Monitor index {self.monitor_index} not found.")
                self.running = False
                return
            seconds = 0
            last_frame = None

            window = gw.getActiveWindow()
            if window:
                wx, wy = window.center

                # Check X bounds
                in_x = monitor["left"] <= wx < (monitor["left"] + monitor["width"])
                # Check Y bounds
                in_y = monitor["top"] <= wy < (monitor["top"] + monitor["height"])

                if in_x and in_y:
                  if window.title not in self.current_minute_apps:
                    self.current_minute_apps.append(window.title)

            while self.running:
                # Capture the screen
                screenshot = sct.grab(monitor)

                # Convert to numpy array
                img = np.array(screenshot)

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

                # Resize if target resolution is specified
                if self.target_resolution:
                    img = cv2.resize(img, self.target_resolution, interpolation=cv2.INTER_AREA)

                # Only add frame if it's different enough from the last one
                if not self._are_images_similar(last_frame, img):
                    with self.lock:
                        self.frame_buffer.append(img)
                    last_frame = img
                    print(f'buffer = {len(self.frame_buffer)} (new frame added)')

                if seconds == 60:
                    with self.lock:
                        # Only process if there's enough new activity
                        if self.activity_logger is not None and len(self.frame_buffer) > 2:
                            # Create a copy for the thread to prevent race conditions
                            frames_to_process = list(self.frame_buffer)
                            overlap_frames = frames_to_process[-2:]
                            self.frame_buffer.clear()
                            self.frame_buffer.extend(overlap_frames)
                        else:
                            # Not enough activity, just clear the buffer
                            self.frame_buffer.clear()
                            frames_to_process = []
                    if frames_to_process:
                        with self.describe_thread_lock:
                            self.describe_thread = Thread(target=self._describe_frames_threaded, args=(frames_to_process, time.time()))
                            self.describe_thread.start()
                    seconds = 2
                # Handle dynamic framerates (adjust as needed)
                time.sleep(1.0 / self.fps)
                seconds +=1


    # todo, not sure for now  how i want to implement it, postponed
    def new_activity(self):
        if self.activity_logger is not None:
            self.activity_logger.reset()
            # self.description_history.clear()

    def cleanup(self):
        self.running = False
        self.capture_thread.join()
        if self.describe_thread is not None:
            self.describe_thread.join()

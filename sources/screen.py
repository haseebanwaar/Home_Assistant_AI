import math
import time
from collections import deque
from threading import Thread, Lock
import cv2
import torch
from PIL import Image
from mss import mss
import numpy as np
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from providers.local_openAI import client, model_name_vlm
from utils.qwen_preprocess import encode_video


class RealtimeScreenCapture:
    def __init__(self, video_source, window_size=60, fps=1.0, monitor_index=1, target_resolution=None,
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
        self.target_resolution = target_resolution
        self.activity_logger = activity_logger
        self.last_minute_logged = int(time.time() / 60)
        self.describe_thread = None  # Thread to handle description
        self.describe_thread_lock = Lock()  # Lock for the thread
        self.description_history = deque(maxlen=3)  #store last 3 description for context
        # Start frame capture thread
        self.capture_thread = Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()


    def _describe_frames(self, imgs):
        """
        Describe the current minute of activity, using VLM
        """
        # print('here')
        # if len(self.description_history) != 0:
        #     question += f'in addition, here is what happened in the past minutes:\n'
        #     for i, description in enumerate(self.description_history):
        #         question += f'Minute {i - len(self.description_history) + 1}: {description}\n'


        print('here')
        messages = [
            {"role": "system", "content": """
a screen capture of previous minutes from the user computer display is provided, help and augment user experience. dont miss eny detail or event.
            """},
            {"role": "user", "content": [
                # {"type": "text", "text": "this is a recent minute video of screen capture, analyze it and output description as precise as possible."},
                {"type": "text", "text": """
Spotting all the text in the image with line-level, and output in JSON format.
                """},
                {'type': 'video_url', 'video_url': {
                    'url': f'data:video/jpeg;base64,{encode_video(torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2), imgs[0].shape[0],imgs[0].shape[1])}'}}] #{','.join(base64_frames)}
             },
        ]
# torch.tensor(video).permute(0, 3, 1, 2)
        tim = time.perf_counter()
        response = client.chat.completions.create(model=model_name_vlm, messages=messages,    extra_body={
            "mm_processor_kwargs": {'fps': [self.fps]}
        }  ,temperature=0.7,max_tokens=2000)
        # } ,temperature=0.7,max_tokens=500)


        answer = response.choices[0].dict()['message']['content']
        print(answer)
        print(time.perf_counter()-tim)



        return answer

    def _describe_frames_internvl(self, imgs):
        """
        Describe the current minute of activity, using VLM
        """
        if len(imgs) == 0:
            return ''

        question = ''
        for i in range(len(imgs)):
            question = question + f'Frame{i + 1}: {IMAGE_TOKEN}\n'
        if len(self.description_history) != 0:
            question += f'in addition, here is what happened in the past minutes:\n'
            for i, description in enumerate(self.description_history):
                question += f'Minute {i - len(self.description_history) + 1}: {description}\n'
        question += 'what do you see?'

        content = [{'type': 'text', 'text': question}]
        for img in imgs:
            content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 9,
                                                               'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})
            # content.append({'type': 'image_url', 'video_url': {
            #                                                    'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})
        messages = [dict(role='user', content=content)]
        tim = time.perf_counter()
        response = client.chat.completions.create(model=model_name_vlm, messages=messages, temperature=0.9,max_tokens=5000)


        answer = response.choices[0].dict()['message']['content']
        print(answer)
        print(time.perf_counter()-tim)


        return answer

    def _describe_frames_threaded(self, imgs, timestamp):
        """
          This method is now threaded.
        """
        print(f'buffer = {timestamp} running')
        try:
            description = self._describe_frames(imgs)
            self.activity_logger.log_activity(description, timestamp)
            with self.lock:
                self.description_history.append(description)
        except Exception as e:
            print(f"An error occurred in _describe_frames_threaded: {e}")
        print(f'buffer = {timestamp} ended')

    def _capture_frames(self):
        frame_count = 0
        with mss() as sct:
            try:
                monitor = sct.monitors[self.monitor_index]
            except IndexError:
                print(f"Error: Monitor index {self.monitor_index} not found.")
                self.running = False
                return
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

                # InternVL requires pil
                # pil_image = Image.fromarray(img)

                timestamp = time.time()
                with self.lock:
                    self.frame_buffer.append(img)
                print(f'buffer = {len(self.frame_buffer)}')
                # Log each minute of activity
                current_minute = int(timestamp / 60)
                if current_minute != self.last_minute_logged:
                    self.last_minute_logged = current_minute
                    if self.activity_logger is not None:
                        with self.describe_thread_lock:
                            self.describe_thread = Thread(target=self._describe_frames_threaded,
                                                          args=(list(self.frame_buffer), timestamp))
                            self.describe_thread.daemon = True
                            self.describe_thread.start()
                        time.sleep(1000.0 / self.fps)

                # Handle dynamic framerates (adjust as needed)
                time.sleep(1.0 / self.fps)
                frame_count += 1

    def new_activity(self):
        if self.activity_logger is not None:
            self.activity_logger.reset()
            self.description_history.clear()

    def cleanup(self):
        self.running = False
        self.capture_thread.join()
        if self.describe_thread is not None:
            self.describe_thread.join()

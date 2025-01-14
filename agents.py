import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
import json
import numpy as np
from lmdeploy import pipeline, GenerationConfig
from decord import VideoReader, cpu
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from PIL import Image
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import pandas as pd
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import BraveSearch
from langchain_community.utilities import SearxSearchWrapper
from autogen_ext.tools.langchain import LangChainToolAdapter
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from tempfile import TemporaryDirectory
from langchain_community.agent_toolkits import FileManagementToolkit
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from providers.local_openAI import model_llm, model_vlm, model_vlm_local, model_name
from typing import Any, Dict, List
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import cv2
import math
import time
import numpy as np
from collections import deque
import cv2
import time
from threading import Thread, Lock
import numpy as np
from PIL import Image
import io



class RealtimeVideoContext:
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
video_context = RealtimeVideoContext(
    video_source="G:/b.mp4",
    # video_source="E:/tour2.mp4",
    window_size=10,  # Keep last 10 seconds
    fps=1  # 1 frame per second
)

# Example usage:
def live_interaction(user_query):
    """
    reads stream from camera and answer user text queries and questions according to current
    context of video
    :return: str
    """
    imgs = list(video_context.frame_buffer)
    question = ''
    for i in range(len(imgs)):
        question = question + f'Frame{i+1}: {IMAGE_TOKEN}\n'
    question += user_query

    content = [{'type': 'text', 'text': question}]
    for img in imgs:
        content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})
    messages = [dict(role='user', content=content)]
    # response = model_vlm_local.chat.completions.create(model=model_name,messages = messages,temperature=0.8,top_p=0.8)
    response = model_vlm_local.chat.completions.create(model=model_name,messages = messages,temperature=1)
    return response.choices[0].dict()['message']['content']

live_interaction('what is happening in this video?')
live_interaction('do you see cars in video?')
live_interaction('how many cars?')
live_interaction('what is happening in this video?')
live_interaction('what is happening?')



persistant one is working 10 sec chunks always
when persistant is working on current 10 sec frame it is made avaialble of last 5 text context
when a live 10 sec request comes last 10 sec frames and text history is attached along with global passivecontext


1-save useful actions from cam or screen all the time with timestamps (an agent can decide if current event is significant
or to be skipped based on previous scope too)
2-need to contain the current scope precisely, if watching, video scope of full video and interactions while watching
should be availble while it lasts. same for cam events timstamped boundaries
3-save timestamped summaries with embeddings and also plain video or images or even audio









# Index summaries by time and keywords for efficient retrieval.
#     Use a vector database (e.g., Pinecone, Weaviate) for storing embeddings.
#     Retrieval:
#
# When a user asks a new query, retrieve summaries that are:
# Close in time to the current query.
# Relevant based on textual or embedding similarity.









planner = AssistantAgent(
    "planner",
    model_client=model,
    handoffs=["camera_balcony", "camera_garage", "camera_front", "web_search", "home_lights"],
    system_message="""
You are a Home Assistant Planning Coordinator.
Your role is to coordinate home monitoring and automation by delegating tasks to specialized agents:
- camera_balcony: Analyzes the video stream from the balcony camera.
- camera_garage: Analyzes the video stream from the garage camera.
- camera_front: Analyzes the video stream from the front door camera.
- web_search: Gathers real-time information from the internet.
- home_lights: Controls the home lights (ON/OFF, color changes).

Always send your plan first, then handoff to appropriate agent.
Always handoff to a single agent at a time.
Use TERMINATE when research is complete.""",
)

camera_garage = AssistantAgent(
    "camera_garage",
    model_client=model_llm,
    handoffs=["planner"],
    tools=[Live_camera_balcony],
    system_message="""
You are a VLM Agent monitoring the garage camera feed.
Your responsibilities:
- Detect human presence
- Monitor for suspicious activity
- Check for weather-related issues (rain, snow)
- Identify if doors/windows are open/closed
- Monitor plants and furniture status

Respond with:
OBSERVATION: [What you see in the feed]
STATUS: [Normal/Alert/Warning]
DETAILS: [Specific information about the observation]""",
)

camera_balcony = AssistantAgent(
    "camera_balcony",
    model_client=modelvl,
    handoffs=["planner"],
    tools=[Live_camera_balcony],
    system_message="""
You are a VLM Agent monitoring the balcony camera feed.
Your responsibilities:
- Detect human presence
- Monitor for suspicious activity
- Check for weather-related issues (rain, snow)
- Identify if doors/windows are open/closed
- Monitor plants and furniture status

Respond with:
OBSERVATION: [What you see in the feed]
STATUS: [Normal/Alert/Warning]
DETAILS: [Specific information about the observation]""",
)

camera_front = AssistantAgent(
    "camera_front",
    model_client=modelvl,
    handoffs=["planner"],
    tools=[Live_camera_balcony],
    system_message="""
You are a VLM Agent monitoring the Front camera feed.
Your responsibilities:
- Detect human presence
- Monitor for suspicious activity
- Check for weather-related issues (rain, snow)
- Identify if doors/windows are open/closed
- Monitor plants and furniture status

Respond with:
OBSERVATION: [What you see in the feed]
STATUS: [Normal/Alert/Warning]
DETAILS: [Specific information about the observation]""",
)



web_search = AssistantAgent(
    "web_search",
    model_client=model,
    # handoffs=["planner"],
    tools=[LangChainToolAdapter(DuckDuckGoSearchRun())],
    system_message="""
You are a Web Search Agent gathering real-time information.
Your responsibilities:
- Monitor local news,weather conditions and alerts
- Search for relevant safety/security updates
- Gather information requested by the planning agent

Respond with:
FINDINGS: [Summary of information found]
SOURCE: [Where the information came from]
RELEVANCE: [How it relates to the request]""",
)



home_lights = AssistantAgent(
    "home_lights",
    model_client=model,
    handoffs=["planner"],
    tools=[Live_camera_balcony],
    system_message="""
You are a Home Lights Control Agent.
Your responsibilities:
- Control light status (ON/OFF)
- Adjust light colors when supported
- Manage lighting scenes
- Execute lighting schedules
- Confirm action completion

Respond with:
ACTION_TAKEN: [What change was made]
AFFECTED_LIGHTS: [Which lights were modified]
STATUS: [Success/Failure]""",
)


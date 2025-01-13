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


model = OpenAIChatCompletionClient(
    # model="F:\\try\\qwen\\Qwen2.5-1.5B-Instruct-AWQ",
    model="F:\\try\\qwen\\Qwen2.5-7B-Instruct-AWQ",
    base_url="http://localhost:23333/v1",
    api_key="placeholder",
    model_capabilities={
        "vision": True,
        "function_calling": True,
        "json_output": True,
    },
)
# Define a tool
modelvl = OpenAIChatCompletionClient(
    # model="F:\\try\\qwen\\Qwen2.5-1.5B-Instruct-AWQ",
    model="F:\\try\\qwen\\InternVL2_5-8B-AWQ",
    base_url="http://localhost:23334/v1",
    api_key="placeholder",
    model_capabilities={
        "vision": True,
        "function_calling": True,
        "json_output": True,
    },
)


async def add_two_numbers(a: int,b: int) -> int:
    return a+b

async def multiply_two_numbers(a: int,b: int) -> int:
    return a*b


async def main() -> None:
    # Define an agent
    weather_agent = AssistantAgent(
        name="weather_agent",
        model_client=model,
        tools=[multiply_two_numbers,add_two_numbers],
    )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")
    # termination = TextMentionTermination("STOP")

    # Define a team
    agent_team = RoundRobinGroupChat([weather_agent], termination_condition=termination, max_turns=12)

    # Run the team and stream messages to the console
    stream = agent_team.run_stream(task="What 5 plus 6?")
    await Console(stream)

asyncio.run(main())








# todo, speedtest local models
import time
tim = time.perf_counter()
agent = AssistantAgent(
    name="weather_agent",
    model_client=model,
)
tokens = 0
async def assistant_run() -> None:
    global tokens
    response = await agent.on_messages(
        [TextMessage(content="what the difference and similarities between tiger, lion. provide a 500 word answer", source="user")],
        cancellation_token=CancellationToken(),
    )
    # print(response.inner_messages)
    tokens= response.chat_message.models_usage.completion_tokens

asyncio.run(assistant_run())
print(tokens/(time.perf_counter() - tim))
















import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

primary_agent = AssistantAgent(
    "primary",
    model_client=model,
    system_message="You are a helpful AI assistant.",
)

# Create the critic agent.
critic_agent = AssistantAgent(
    "critic",
    model_client=model,
    system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)


result = asyncio.run( team.run(task="Write a short poem about the fall season."))
print(result)













from typing import Any, Dict, List
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


def refund_flight(flight_id: str) -> str:
    """Refund a flight"""
    return f"Flight {flight_id} refunded"


travel_agent = AssistantAgent(
    "travel_agent",
    model_client=model,
    handoffs=["flights_refunder", "user"],
    system_message="""You are a travel agent.
    The flights_refunder is in charge of refunding flights.
    If you need information from the user, you must first send your message, then you can handoff to the user.
    Use TERMINATE when the travel planning is complete.""",
)

flights_refunder = AssistantAgent(
    "flights_refunder",
    model_client=model,
    handoffs=["travel_agent", "user"],
    tools=[refund_flight],
    system_message="""You are an agent specialized in refunding flights.
    You only need flight reference numbers to refund a flight.
    You have the ability to refund a flight using the refund_flight tool.
    If you need information from the user, you must first send your message, then you can handoff to the user.
    When the transaction is complete, handoff to the travel agent to finalize.""",
)



termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")
team = Swarm([travel_agent, flights_refunder], termination_condition=termination)

task = "can you help me refund my flight."


async def run_team_stream() -> None:
    task_result = await Console(team.run_stream(task=task))
    last_message = task_result.messages[-1]

    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(
            team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]


asyncio.run( run_team_stream())










working_directory = TemporaryDirectory(dir =r'F:\temp')
toolkit = FileManagementToolkit(
    root_dir=str(working_directory.name)
)  # If you don't provide a root_dir, operations will default to the current working directory
l = toolkit.get_tools()


async def main() -> None:
    tool = [LangChainToolAdapter(i) for i in l]
    tool = [*tool,LangChainToolAdapter(DuckDuckGoSearchRun())]
    model_client = model
    agent = AssistantAgent(
        "assistant",
        # tools=[tool],
        tools=tool,
        model_client=model_client,
        system_message="You are an AI assistant with web search and file handling capabilities. Your task is to perform web searches and save results systematically.",
    )
    await Console(
        web_search.on_messages_stream(
            # [TextMessage(content="1- Search the web for Allama Iqbal.\n2- Create a new file locally with name iqbal.txt.\n3- write the resulting text of search in iqbal.txt", source="user")], CancellationToken()
            [TextMessage(content="""search the web for elon""", source="user")], CancellationToken()
        )
    )
asyncio.run(main())

async def main() -> None:
    model_client = model
    agent = AssistantAgent(
        "assistant",
        tools=[tool],
        model_client=model_client,
        system_message="You are an AI assistant.",
    )
    await Console(
        agent.on_messages_stream(
            # [TextMessage(content="1- Search the web for Allama Iqbal.\n2- Create a new file locally with name iqbal.txt.\n3- write the resulting text of search in iqbal.txt", source="user")], CancellationToken()
            [TextMessage(content=content, source="user")], CancellationToken()
        )
    )
asyncio.run(main())









def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    imgs = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        imgs.append(img)
    return imgs




def Live_camera_balcony(max_frames):

    video_path = 'e:/tour2.mp4'
    imgs = load_video(video_path, num_segments=max_frames)

    question = ''
    for i in range(len(imgs)):
        question = question + f'Frame{i+1}: {IMAGE_TOKEN}\n'

    question += 'describe this video in maximum detail. describe each segment and settings?'

    content = [{'type': 'text', 'text': question}]
    for img in imgs:
        content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})

    messages = [dict(role='user', content=content)]
    out = pipe(messages, gen_config=GenerationConfig(top_k=1))
    out = pipe(messages, gen_config=gen_config)
    out.text


import cv2
import math
import time
import numpy as np

def extract_frames_per_second(mp4_file_path,sfps):
    try:
        cap = cv2.VideoCapture(mp4_file_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file '{mp4_file_path}'.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Get frames per second
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # Get frames per second
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Get frames per second
        duration =math.ceil(total_frames/fps)  # Get frames per second

        frame_count = 0
        seconds_passed = 0

        while True:
            ret, frame = cap.read()

            if not ret:  # End of the video
                break

            frame_count += 1

            # Check if one second has passed
            if frame_count >= (seconds_passed + 1) * sfps:
                seconds_passed = seconds_passed+ (sfps/fps)
                time.sleep(sfps/fps)
                yield frame

        cap.release()

    except Exception as e:
        print(f"An error occurred: {e}")
        yield None


# Example usage:
for i,frame in enumerate(extract_frames_per_second("e:/tour2.mp4",10)):
    if frame is not None:
        print(i)
        # cv2.imwrite(f"./frames/{i}.jpg", frame)






from collections import deque
import cv2
import time
from threading import Thread, Lock
import numpy as np
from PIL import Image
import io


from openai import OpenAI
client = OpenAI(api_key='ss', base_url='http://localhost:23334/v1')
model_name = client.models.list().data[0].id

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

            if frame_count % (30 // self.fps) == 0:  # Assuming 30fps video
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
    video_source="E:/tour2.mp4",
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
    question += 'what am i trying to capture while vlogging?'

    content = [{'type': 'text', 'text': question}]
    for img in imgs:
        content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})
    messages = [dict(role='user', content=content)]

    # out = pipe(messages, gen_config=GenerationConfig(top_k=1))



response = client.chat.completions.create(
    model=model_name,
    messages = messages,
    temperature=0.8,
    top_p=0.8)
print(response)



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
    model_client=modelvl,
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


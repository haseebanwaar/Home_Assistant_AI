from lmdeploy.vl.constants import IMAGE_TOKEN
import time
from collections import deque

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.tools import DuckDuckGoSearchRun
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from PIL import Image

from providers.local_openAI import  client,model_name_vlm
from sources.screen import RealtimeScreenCapture
from sources.file import RealtimeFileContext
from vecttor_store.activity_logger import ActivityLogger
# Initialize Activity Logger
activity_logger = ActivityLogger()

# Initialize screen capture
screen_stream = RealtimeScreenCapture(
    video_source="",  # Not used
    window_size=60,
    fps=1,  # Capture 2 frames per second
    monitor_index=2, # Capture the primary monitor
    activity_logger=activity_logger # we pass activity logger here
    # target_resolution=(1720, 720), # Reduce resolution to 1280x720
    # target_resolution=(720, 1280), # Reduce resolution to 1280x720
)

# Initialize screen capture
# file_stream = RealtimeFileContext(
#     video_source=r"C:\d\b.mp4",  # Not used
#     window_size=60,
#     fps=1,  # Capture 2 frames per second
#
#     # target_resolution=(1720, 720), # Reduce resolution to 1280x720
#     # target_resolution=(720, 1280), # Reduce resolution to 1280x720
# )

# Global list to maintain the chat history
chat_history = []

def live_interaction(video_context, user_query,previous_content=None):
    """
    Reads stream from camera and answer user text queries and questions according to current
    context of video
    :return: str
    """
    global chat_history  # Access the global chat history

    # Clear frame_buffer if needed
    #video_context.frame_buffer.clear()

    imgs = list(video_context.frame_buffer)

    question = ''
    for i in range(len(imgs)):
        question = question + f'Frame{i+1}: {IMAGE_TOKEN}\n'
    # Search for relevant past activity
    # past_activity = activity_logger.search_activity(user_query)
    # if len(past_activity) != 0:
    #     question = question + "in addition, here is what happened in the past :\n"
    #     for activity,metadata in past_activity:
    #         question=question+f"at {metadata['timestamp']}, description:{activity}\n"
    question += user_query  # Use the user's query directly

    content = [{'type': 'text', 'text': question}]
    for img in imgs:
        content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 10, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})
    messages = []
    if previous_content is not None:
        messages = previous_content

    messages.append(dict(role='user', content=content))

    # response = model_vlm.chat.completions.create(model=model_name_vlm,messages = messages,temperature=0.5,top_p=0.95,max_tokens=64000)
    # response = model_vlm.chat.completions.create(model=model_name_vlm,messages = messages,temperature=0.95,top_p=0.95,max_tokens=4000)
    response = client.chat.completions.create(model=model_name_vlm,messages = messages,temperature=0.7,max_tokens=4000)

    answer = response.choices[0].dict()['message']['content']

    # Append current interaction to chat history
    messages.append(response.choices[0].dict()['message'])

    chat_history = messages

    print(answer)
    return answer


def ask_follow_up_question(video_context, follow_up_query):
    """Asks a follow-up question, maintaining conversation history."""
    global chat_history
    return live_interaction(video_context, follow_up_query, chat_history)
def start_new_activity(video_context):
    """Manually trigger a new activity."""
    video_context.new_activity()
# Initial interaction
time.sleep(5000) #wait for 1 minute

live_interaction(screen_stream,'what is happening here')
# Follow-up questions
ask_follow_up_question(screen_stream,'what was i doing in this video, around 1 minute ago?')
ask_follow_up_question(screen_stream,'can you tell what i am doing in this video?')
ask_follow_up_question(screen_stream,'can you write the code you see on the screen?')
ask_follow_up_question(screen_stream,'focus on extreme right where you see green graphs. explain each sub graphs?')
ask_follow_up_question(screen_stream,'what is happening?')
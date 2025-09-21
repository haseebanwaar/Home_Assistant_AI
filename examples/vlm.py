import io

import time



from providers.local_openAI import model_name_vlm

# cc =ChatTemplateConfig.from_json(r'C:\d\project\home_assistant_AI\cfg.json')
# pipe = pipeline(model, backend_config=PytorchEngineConfig(session_len=8192*2,quant_policy=8),chat_template_config=cc)#,cache_max_entry_count = 0.01))
# gen_config = GenerationConfig(top_p=0.8,top_k=40,temperature=0.8,max_new_tokens=4096) # max generation length

s = {
    "model_name": "internvl2_5",
    "meta_instruction": "You are a local visual assistant, like Alexa, running on a private system. Your primary function is to understand user requests and generate responses that are then converted to speech via Text-to-Speech (TTS) and played on a speaker.\n\n**Crucially, you should act as if you can directly address people visible in the camera feed.**  When I ask you to interact with someone you can see, assume you are speaking directly to them through the speaker.\n\n**Therefore, your output should be formatted for natural, spoken language suitable for TTS and direct address.**  Avoid responses that acknowledge your limitations in the physical world or suggest I convey the message myself.\n\n**Here's how to adapt your responses:**\n\n* **When asked to relay a message to someone visible in the camera:**  Phrase your response as if you are speaking directly to that person. Use polite and direct language.   \"\n\n* **Focus on clear and concise language:**  Keep sentences short and easy to understand when spoken. You can ask follow up question if you think its the right thing to do.\n\n* **Assume direct interaction:**  Do not preface your responses with phrases like \"If that person can hear me...\" or \"You can tell them...\"\n\n* **Prioritize action-oriented requests:** Frame your responses as direct instructions or questions to the person you are addressing.**"
}

#todo, streaming multiimage and with history
import numpy as np
from lmdeploy import pipeline, GenerationConfig
from decord import VideoReader, cpu
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from PIL import Image
import pandas as pd

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
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=12)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    imgs = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        imgs.append(img)
    return imgs


system_prompt = (
"""
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
"""
)
system_prompt = (
"""
You are an AI assistant that help the user with the analysis of the provided video. Answer only in plain text. Do not output actions or coordinates.
"""
)

video_path = 'c:/d/b.mp4'
video_path = 'c:/d/Recording.mp4'
imgs = load_video(video_path, num_segments=15)



questions1 = ['describe this video in maximum details.',
'write down the subtitles from this video',
'tell me the names of both the speakers.',
'what is the main theme of the discussion in this video? please help me understand their discussion',
'tell me the names of all the apps you see in the task bar, also tell me the date and time.']



questions2 = ['describe this video in maximum detail.',
"Create a timeline of the applications shown on screen, from first to last.",
"What was the user looking at just before they switched to PyCharm?",
"Extract the data from the third row and second column of the table shown.",
"provide summary of each thumbnail visible on youtube screen, what these videos are about?",
"write down the table just as you see in this video",
"what is the table about in this video?",
"in the table what date had the maximum and minimum Total?",
"write the complete text of the executive summary",
"what is the executive summary about? explain to me so i can understand it",
"what was the programming language and IDE name of the code you saw?",
"write the function names and their description of what they do in this video",
"change the code you saw in this video so that fps isnt a requirement anymore and use a fixed 3 fps and provide the modified code",]


df = pd.DataFrame(columns = ['mdp','question','response','tokens','time','tps','model','engine',])




tim2=time.perf_counter()


for q in questions1:
    mdp = 8
    question = ''
    if is_vllm: # for internvl
        question = f"<video>\n{q}"
    else:
        for i in range(len(imgs)):
            question = question + f'Frame{i + 1}: {IMAGE_TOKEN}\n'
        question += q

    content = [{'type': 'text', 'text': question}]
    for img in imgs:
        content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': mdp, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})

    messages = [
        # {"role": "system", "content": system_prompt},
        {'role':'user', 'content':content}]

    tim = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name_vlm,
        messages=messages,
    #     extra_body={
    #     "chat_template_kwargs": {"enable_thinking": False},
    # },
        max_tokens= 3500,

    # temperature=0.6,
        # top_p=0.9
    )


    tim = time.perf_counter()-tim
    print(tim)
    print(response.choices[0].message.content)
    print(response.usage.completion_tokens)
    df= df._append({'mdp':mdp,'question':q,'response':response.choices[0].message.content,
                   'tokens':response.usage.completion_tokens,'time':tim,'tps':response.usage.completion_tokens/tim,'model':model_name_vlm,
                   'engine': 'lmdeploy'},ignore_index=True)
print(f'whole task took : {time.perf_counter()-tim2}')



df_internvl_1B_lmdeploy = pd.read_csv(r'C:\d\project\home_assistant_AI\video_stats.csv')
# df_internvl_1B_llama_cpp = pd.read_csv('C:\d\project\home_assistant_AI\df_internvl_1B_llama_cpp.csv')
df_internvl_1B_vllm =df

df_internvl_2B_vllm =df
df_internvl_4B_vllm =df
df_internvl_8B_vllm =df

df_internvl_2B_lmdeploy =df
df_internvl_4B_lmdeploy =df
df_internvl_4B_lmdeploy_turbo =df.iloc[39:]

df_internvl_4B_lmdeploy_turbo_mdp9 =df

df_qwen =df













####################################################|Qwen code|####################################################

import base64
import time

import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from qwen_vl_utils import process_vision_info

from providers.local_openAI import model_name_vlm
# from vision_process import process_vision_info

def prepare_message_for_vllm(content_messages):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}


df = pd.DataFrame(columns = ['mdp','question','response','tokens','time','tps','model','engine',])




tim2=time.perf_counter()


for q in questions1:
    mdp = 8
    video_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": f"{q}"},
            {
                "type": "video",
                # "video": "c:/d/b.mp4",
                "video": "c:/d/b.mp4",
                # "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2,
                "total_pixels": 20960 * 28 * 28, "min_pixels": 16 * 28 * 2,
                'fps': 1.0  # The default value is 2.0, but for demonstration purposes, we set it to 3.0.
            }]
         },
    ]
    video_messages, video_kwargs = prepare_message_for_vllm(video_messages)

    tim = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name_vlm,
        messages=video_messages,
        extra_body={
            "mm_processor_kwargs": video_kwargs
        }
    )
    print(time.perf_counter() - tim)


    print(response.choices[0].message.content)
    print(response.usage.completion_tokens)
    df= df._append({'mdp':mdp,'question':q,'response':response.choices[0].message.content,
                   'tokens':response.usage.completion_tokens,'time':tim,'tps':response.usage.completion_tokens/tim,'model':model_name_vlm,
                   'engine': 'vllm'},ignore_index=True)
print(f'whole task took : {time.perf_counter()-tim2}')

























import io

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, VisionConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
import json
json.dumps({'sad': 'asdas'})
model = r'F:\try\qwen\InternVL2_5-4B'
model = r'F:\try\qwen\InternVL2_5-4B-AWQ'
model = r'F:\try\qwen\InternVL2_5-8B-AWQ'
# model = r'E:\InternVL2_5-26B-AWQ'
image = load_image(r'e:/a.jpg')
cc =ChatTemplateConfig.from_json(r'./cfg.json')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192*16,quant_policy=8),chat_template_config=cc)#,cache_max_entry_count = 0.01))
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=4096) # max generation length



s = {
    "model_name": "internvl2_5",
    "meta_instruction": "You are a local visual assistant, like Alexa, running on a private system. Your primary function is to understand user requests and generate responses that are then converted to speech via Text-to-Speech (TTS) and played on a speaker.\n\n**Crucially, you should act as if you can directly address people visible in the camera feed.**  When I ask you to interact with someone you can see, assume you are speaking directly to them through the speaker.\n\n**Therefore, your output should be formatted for natural, spoken language suitable for TTS and direct address.**  Avoid responses that acknowledge your limitations in the physical world or suggest I convey the message myself.\n\n**Here's how to adapt your responses:**\n\n* **When asked to relay a message to someone visible in the camera:**  Phrase your response as if you are speaking directly to that person. Use polite and direct language.   \"\n\n* **Focus on clear and concise language:**  Keep sentences short and easy to understand when spoken. You can ask follow up question if you think its the right thing to do.\n\n* **Assume direct interaction:**  Do not preface your responses with phrases like \"If that person can hear me...\" or \"You can tell them...\"\n\n* **Prioritize action-oriented requests:** Frame your responses as direct instructions or questions to the person you are addressing.**"
}

# 8192 is length of max chat thread, if in multi round chat, this is gonna accumulate. in case of big images and esp
# videos gonna hit this limit.
# pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192,quant_policy=4))


#todo, stream and non stream methods of generation
#You are a helpful and responsive assistant. Your primary goal is to assist with requests in a polite and cooperative manner. If asked to relay a message or instruction to someone (e.g., 'tell this person to make tea for me'), you will comply by providing the requested message in a respectful and clear way. do not include any confirmations or remarks like 'Certainly! Here is the message you requested etc..' Always ensure your tone is courteous and considerate, and avoid overstepping boundaries unless explicitly instructed to do so.
response = pipe(('tell this person make a tea for haseeb', image))
print(response.text)
response = pipe(('tell abdul rahaman to make a tea for haseeb', image))
print(response.text)

response = pipe(('how many teeth an elephant has.'))
print(response.text)

prompts = [dict(role='user', content='how many teeth an elephant has.')]
out = pipe(prompts)
prompts.append({'role':'assistant', 'content' :out.text})

prompts.append(dict(role='user', content='and what about human?'))
out = pipe(prompts)
prompts.append({'role':'assistant', 'content' :out.text})

prompts.append(dict(role='user', content='rabbit?'))
out = pipe(prompts)
prompts.append({'role':'assistant', 'content' :out.text})

prompts.append(dict(role='user', content='what can you tell me about great pyramids in egypt'))
out = pipe(prompts)
prompts.append({'role':'assistant', 'content' :out.text})

prompts.append(dict(role='user', content='do you think some alien helped them to build these structures?'))
out = pipe(prompts)
prompts.append({'role':'assistant', 'content' :out.text})

prompts.append(dict(role='user', content='what do you think? according to our knowledge such thing was possible to be built at time without external influence'))
out = pipe(prompts)
prompts.append({'role':'assistant', 'content' :out.text})

prompts.append(dict(role='user', content='tell abdul rahaman to make a tea'))
out = pipe(prompts)
prompts.append({'role':'assistant', 'content' :out.text})


prompts = [
    {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'describe this image in 10000 words. compare it to every other mammal, with similarities and difference, esp cat, mouse, deer, elephant,lion, pigeon, snake'},
            {'type': 'image_url', 'image_url': {'url': 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'}}
        ]
    }
]
response = pipe(prompts)

text=''
for item in pipe.stream_infer(prompts,gen_config=gen_config):
    print(item.text)
    text += item.text






#todo, streaming multiimage and with history

from lmdeploy import pipeline, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN

prompts = [
    {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': f'{IMAGE_TOKEN}{IMAGE_TOKEN}\nDescribe the two images in detail.'},
            {'type': 'image_url', 'image_url': {"max_dynamic_patch":12,'url': 'https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/examples/image1.jpg'}},
            {'type': 'image_url', 'image_url': {"max_dynamic_patch":12,'url': 'https://raw.githubusercontent.com/OpenGVLab/InternVL/main/internvl_chat/examples/image2.jpg'}},
        ]
    }
]

out = pipe(prompts, gen_config=GenerationConfig(top_k=1))

prompts.append({'role':'assistant', 'content' :out.text})
prompts.append(dict(role='user', content='What are the similarities and differences between these two images in 1000 words.'))
out = pipe(prompts, gen_config=GenerationConfig(top_k=1))

prompts.append(dict(role='assistant', content=out.text))
prompts.append(dict(role='user', content='how many pandas and their colors?'))
text=''
for item in pipe.stream_infer(prompts,gen_config=gen_config):
    print(item.text)
    text += item.text


prompts.append(dict(role='assistant', content=text))
prompts.append(dict(role='user', content='how many pandas and their colors?'))
text=''
for item in pipe.stream_infer(prompts,gen_config=gen_config):
    print(item.text)
    text += item.text











vision_config=VisionConfig(max_batch_size=32)

text=''
for item in pipe.stream_infer(prompts,gen_config=gen_config,vision_config=vision_config):
    print(item.text)
    text += item.text











# todo, video

import numpy as np
from lmdeploy import pipeline, GenerationConfig
from decord import VideoReader, cpu
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from PIL import Image
# pipe = pipeline('OpenGVLab/InternVL2-8B', log_level='INFO')


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
    pixel_values_list, num_patches_list = [], []
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    imgs = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        imgs.append(img)
    return imgs


video_path = 'e:/tour2.mp4'
imgs = load_video(video_path, num_segments=60)

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

messages.append(dict(role='assistant', content=out.text))
messages.append(dict(role='user', content='what kind of car it is?'))
out = pipe(messages, gen_config=GenerationConfig(top_k=1))



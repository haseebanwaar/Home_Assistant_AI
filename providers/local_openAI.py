from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import OpenAI
import openai
import json

# C:\d\VLM\videovenv\Scripts\activate
# llama-server -m "F:\try\intern_s1_gguf\Intern-S1-mini-Q8_0.gguf" --mmproj "F:\try\intern_s1_gguf\mmproj-Intern-S1-mini-Q8_0.gguf" -ngl 100 --temp 0.8 --top-p 0.8 --top-k 50

# llama-server -m c:/d/orpheus-3b-0.1-ft-q2_k.gguf -ngl 100 -c 2048


# lmdeploy serve api_server D:\VLM\qwen\InternVL2_5-8B-AWQ --server-port 23334 --cache-max-entry-count 0.2 --session-len 9999 --quant-policy 4
# lmdeploy serve api_server D:\VLM\qwen\Qwen2.5-7B-Instruct-AWQ --server-port 23333 --cache-max-entry-count 0.2 --session-len 9999 --quant-policy 4
# lmdeploy serve api_server C:\d\project\home_assistant_AI\intern1-W4A16-G128 --server-port 23333 --cache-max-entry-count 0.2 --session-len 9999
# lmdeploy serve api_server F:\try\internVL_1B --server-port 23333 --cache-max-entry-count 0.3 --session-len 44000
# lmdeploy serve api_server F:\try\intern_s1 --server-port 23333 --cache-max-entry-count 0.2 --session-len 9999 --quant-policy 4

# source /home/haseeb/venv/bin/activate
# vllm serve "/mnt/f/try/internVL_1B" --max-model-len 40000 --gpu-memory-utilization 0.1 --trust-remote-code
# vllm serve "/mnt/f/try/intern_s1" --max-model-len 4000 --gpu-memory-utilization 0.1 --trust-remote-code
# vllm serve "hfl/Qwen2.5-VL-7B-Instruct-GPTQ-Int4"  --max-model-len 40000 --gpu-memory-utilization 0.75 --max_num_seqs 1



#
client = OpenAI(api_key='ss', base_url='http://localhost:23333/v1')
model_name_vlm = client.models.list().data[0].id
#
# client = OpenAI(api_key='ss', base_url='http://localhost:8080/v1')
# model_name_vlm = client.models.list().data[0].id


client = OpenAI(api_key='ss', base_url='http://localhost:8000/v1')
model_name_vlm = client.models.list().data[0].id


#
# import base64,os, time
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')
#
#
# # # Example usage:
# image_path = "c:/d/a.png"
# base64_image = encode_image(image_path)
# #
#
#

import time
tim = time.perf_counter()
chat_response = client.chat.completions.create(
    model= model_name_vlm,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        'url': f'data:image/jpeg;base64,{base64_image}'              },
                },
                {"type": "text", "text": "what do you see in this image?"},
            ],
        },
    ],
)
print("Chat response:", chat_response.model_dump()['choices'][0]['message']['content'])
print(time.perf_counter()-tim)













import asyncio
import base64
from http.client import HTTPException
from typing import Dict, List, Optional, Union
import cv2
import nest_asyncio
from openai import BaseModel
import pydub
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.encoders import jsonable_encoder
import librosa
import io
from lmdeploy import pipeline
from lmdeploy.vl import load_image
from PIL import Image
import threading
from lmdeploy import pipeline, TurbomindEngineConfig
from pydub import AudioSegment
from qwen_agent.llm import get_chat_model
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import uvicorn
import io
import soundfile as sf
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import wave
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoProcessor, Wav2Vec2ForCTC

from faster_whisper import WhisperModel
from qwen_agent.llm import get_chat_model
from qwen_agent.llm.schema import ContentItem
from qwen_agent.utils.utils import save_url_to_local_work_dir
import json
import os
import urllib.parse

import torch
from faster_whisper import WhisperModel

import librosa
import pydub
nest_asyncio.apply()


app = FastAPI()





buffer_dict: Dict[str, io.BytesIO] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

print('running app')

model = Qwen2VLForConditionalGeneration.from_pretrained(
    r"F:\try\qwen\qwen7b4",
    torch_dtype="auto",
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)
processor = AutoProcessor.from_pretrained(r"F:\try\qwen\qwen7b4")



#------------------response---------------------
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, str]]]

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False

def generate_vlm(messages):
    # Process messages
    # messages[-1].content.extend([{
    #     "type": "image",
    #     "image": "d:/cars.png",
    # }])
    # processed_messages = []
    # for msg in messages:
    #     if isinstance(msg.content, list):
    #         processed_content = []
    #         for content_item in msg.content:
    #             if content_item['type'] == 'image':
    #                 processed_content.append({
    #                     'type': 'image',
    #                     #adding dummy image
    #                     'image': r"d:/cars.png"
    #                     # 'image': content_item['image']
    #                 })
    #             else:
    #                 processed_content.append({
    #                     'type': 'text',
    #                     'text': content_item['text']
    #                 })
    #         processed_messages.append({
    #             'role': msg.role,
    #             'content': processed_content
    #         })
    #     else:
    #         processed_messages.append({
    #             'role': msg.role,
    #             'content': [{'type': 'text', 'text': msg.content}]
    #         })

    # Prepare for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs = process_vision_info(processed_messages)
    inputs = processor(
        text=[text],
        # images=[Image.open(r'd:/cars.png')],
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

async def generate_streaming_response(messages, stop):
    full_response = generate_vlm(messages)
    chunks = full_response.split()
    for chunk in chunks:
        if any(s in chunk for s in (stop or [])):
            break
        yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk + ' '}}]})}\n\n"
        await asyncio.sleep(0.1)  # Simulate delay
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
# async def chat_completions(request: ChatCompletionRequest):
async def chat_completions(request: Request):
    try:
        if request.stream:
            return StreamingResponse(
                generate_streaming_response(request.messages, request.stop),
                media_type="text/event-stream"
            )
        else:
            response = generate_vlm(request.messages)
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response
                        }
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





def turn_off_lights_kitchen(one_or_all: str) -> str:
    return f'{one_or_all} lights are turned off in kitchen'

def turn_off_lights_garage(one_or_all: str) -> str:
    return f'{one_or_all} lights are turned off in garage'



def turn_on_lights_kitchen(one_or_all: str) -> str:
    return f'{one_or_all} lights are turned on in kitchen'


def turn_on_lights_garage(one_or_all: str) -> str:
    return f'{one_or_all} lights are turned on in garage'

functions = [
    {
        'name': 'turn_off_lights_kitchen',
        'description': 'turns off all the lights in kitchen',
        'parameters': {
            'name': 'one_or_all',
            'type': 'string',
            'description': 'number of lights to turn off or turn off all of them',
            'required': True
        }
    },
    {
        'name': 'turn_off_lights_garage',
        'description': 'turns off all the lights in garage',
        'parameters': {
            'name': 'one_or_all',
            'type': 'string',
            'description': 'number of lights to turn off or turn off all of them',
            'required': True
        }
    },
    {
        'name': 'turn_on_lights_kitchen',
        'description': 'turns on all the lights in kitchen',
        'parameters': {
            'name': 'one_or_all',
            'type': 'string',
            'description': 'number of lights to turn on or turn on all of them',
            'required': True
        }
    },
    {
        'name': 'turn_on_lights_garage',
        'description': 'turns on all the lights in garage',
        'parameters': {
            'name': 'one_or_all',
            'type': 'string',
            'description': 'number of lights to turn on or turn on all of them',
            'required': True
        }
    },
]




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
    # uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info",ssl_certfile="cert.crt", ssl_keyfile="cert.key")
    # uvicorn.run("app:app")

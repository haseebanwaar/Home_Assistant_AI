import asyncio
import base64
import uuid
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



modelUrdu = WhisperModel("large-v3", device="cuda", compute_type="int8")
# model = WhisperModel("large-v3", device="cuda", compute_type="float16")
# modelW = WhisperModel('./whisperUrdu', device='cuda', )  # num_workers= 12)
# modelUrdu = WhisperModel("distil-large-v2", device='cuda', compute_type='float16')


vad, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                            model='silero_vad',
                            # source='local',  # )
                            onnx=True)




buffer_dict: Dict[str, io.BytesIO] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_cfg_oai = {
    'model_type': 'qwenvl_oai',
    'model': 'Qwen2-VL-7B-Instruct',
    'model_server': 'http://localhost:8001/v1',  # api_base
    'api_key': 'EMPTY',
}
llm = get_chat_model(llm_cfg_oai)











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







def inference(txt,img):
    messages = [{
        'role':
            'user',
        'content': [ {
            'text': txt
        }]
    }]
    if img is not None:
        messages[-1]['content'].extend([{
            # "image": Image.open(io.BytesIO(base64.b64decode(img))),
            "image": "data:image;bse64," + str(base64.b64decode(img))
            # "image": base64.b64decode(img)
        }])


    print('# Assistant Response 1:')
    responses = []
    for responses in llm.chat(messages=messages, functions=functions, stream=True,extra_generate_cfg=dict(parallel_function_calls=True),):
        print(responses)
    messages.extend(responses)

    ### dig into it why call it back again twice. also no natural language response?
    # for rsp in responses:
    #     if rsp.get('function_call', None):
    #         func_name = rsp['function_call']['name']
    #         if func_name == 'turn_off_lights_kitchen':
    #             func_args = json.loads(rsp['function_call']['arguments'])
    #             image_url = turn_off_lights_kitchen(func_args['one_or_all'])
    #             print('# Function Response:')
    #             func_rsp = {
    #                 'role': 'function',
    #                 'name': func_name,
    #                 'content': [ContentItem(text=image_url)],
    #             }
    #             messages.append(func_rsp)
    #             print(func_rsp)
    #         else:
    #             raise NotImplementedError
    #
    # print('# Assistant Response 2:')
    # responses = []
    # for responses in llm.chat(messages=messages, functions=functions, stream=True):
    #     print(responses)
    # messages.extend(responses)


















@app.post("/transcribe_file/urdu")
async def process_file_urdu(request: Request):
    lang_code = 'urdu'
    wav_bytes = await request.body()
    wav_bytes = wav_bytes.decode('utf-8')
    wav_bytes = json.loads(wav_bytes)
    wav_bytes_audio = wav_bytes['audio']
    # if 'image' in wav_bytes.keys():
    wav_bytes_image = wav_bytes['image'] if 'image' in wav_bytes.keys() else None
    wav_bytes_video = wav_bytes['video'] if 'video' in wav_bytes.keys() else None

    return StreamingResponse(
        generate_flutter(wav_bytes_audio, wav_bytes_image,wav_bytes_video, lang_code),
        media_type="text/plain",
        headers={"Content-type": "text/plain"},
    )


# @profile
async def generate_flutter(wav_bytes_audio, wav_bytes_image, wav_bytes_video, lang):
    print('file')
    # res = bytes(json.loads(wav_bytes_audio.decode("utf-8")))
    # res = bytes(json.loads(wav_bytes_audio))
    res = bytes(wav_bytes_audio)
    a = pydub.AudioSegment.from_wav(io.BytesIO(res))
    sampling_rate = 16000
    if a.frame_rate != sampling_rate:
        a = a.set_frame_rate(sampling_rate)
    if a.channels != 1:
        a = a.set_channels(1)
    (get_speech_timestamps, _, read_audio, *_) = utils

    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(a.channels)  # 1,2
        wav_file.setsampwidth(a.sample_width)  # 2,3,4
        wav_file.setframerate(a.frame_rate)  # 16000,22050, 48000
        wav_file.writeframesraw(a.raw_data)
    res = wav_io.getvalue()

    wav = read_audio(io.BytesIO(res), sampling_rate=sampling_rate)
    speech_timestamps = get_speech_timestamps(wav, vad, sampling_rate=sampling_rate,
                                              visualize_probs=True,  threshold=0.4)

    # speech_timestamps = get_speech_timestamps(wav, vad, sampling_rate=sampling_rate,
    #                                       visualize_probs=True, return_seconds=True, threshold=0.2, min_speech=0.1,
    #                                       min_silence=0.1)


    # a = pydub.AudioSegment.from_file(io.BytesIO(res))
    print(speech_timestamps)
    print(f'total segments: {len(speech_timestamps)}')

    # segments = pydub.silence.split_on_silence(a, 100, silence_thresh=-16)  # Adjust the threshold as needed
    # time_stamp = pydub.silence.detect_nonsilent(a, 100, silence_thresh=-16)  # Adjust the threshold as needed
    for seg in speech_timestamps:
        # temp = a.get_sample_slice(math.trunc(seg['start']), math.ceil(seg['end']))
        temp = a.get_sample_slice(seg['start'], seg['end'])

        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(temp.channels)  # 1,2
            wav_file.setsampwidth(temp.sample_width)  # 2,3,4
            wav_file.setframerate(temp.frame_rate)  # 16000,22050, 48000
            wav_file.writeframesraw(temp.raw_data)
        res = wav_io.getvalue()

        data, samplerate = sf.read(io.BytesIO(res))

        # if lang == 'urduuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu':
        #     # ,vad_parameters=dict(min_silence_duration_ms=200), )
        #     segments, info = modelUrdu.transcribe(data, vad_filter=False,
        #                                        language='ur',
        #                                        without_timestamps=True)
        #     transcription = ''
        #     for segment in segments:
        #         print("\nfrom whisper: [%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        #         transcription += segment.text
        #
        #     full_transcription = milliseconds_to_time(seg) + '|     ' + transcription + '\n'
        #
        # else:
        segments, info = modelUrdu.transcribe(data, vad_filter=False,
                                           language='en',
                                           without_timestamps=True)
        transcription = ''
        for segment in segments:
            print("\nfrom whisper: [%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            transcription += segment.text

    # full_transcription = milliseconds_to_time(seg) + '|     ' + transcription
    full_transcription = transcription
    if wav_bytes_image:

        image_path =rf'C:\vlm_temp\images\{uuid.uuid4()}.jpg'
        cv2.imwrite(image_path,cv2.imd  ecode(np.frombuffer(base64.b64decode(wav_bytes_image),np.uint8),cv2.IMREAD_COLOR))
        # Image.open(io.BytesIO(base64.b64decode(wav_bytes_image))).save(image_path)
    if wav_bytes_video:

        video_path = Image.open(io.BytesIO(base64.b64decode(wav_bytes_video)))

    full_transcription = f'Query: {full_transcription}\nReponse: {inference(full_transcription,image_path,video_path)}'



    yield full_transcription


def milliseconds_to_time(milliseconds):
    seconds = milliseconds['start'] / 16000
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    from_time = f'{str(int(hours)).zfill(2)}:{int(minutes):02}:{seconds:05.2f}'

    seconds = milliseconds['end'] / 16000
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    to_time = f'{str(int(hours)).zfill(2)}:{int(minutes):02}:{seconds:05.2f}'
    return to_time + ' - ' + from_time


# @app.post("/process_multi_files/")
# async def process_file_urdu(request: Request):
#     lang_code = 'urd'
#     # wav_bytes = await request.body()
#     wav_bytes = await request.form()
#     files = wav_bytes.getlist('files')
#     for file in files:
#         filename = file.filename
#         content_type = file.content_type
#         content = await file.read()  # Read the file content as bytes
#
#         uploaded_files.append({
#             "filename": filename,
#             "content_type": content_type,
#             "size": len(content),
#             "content": content
#         })
#     # return StreamingResponse(
#     #     generate(wav_bytes, lang_code),
#     #     media_type="text/plain",
#     #     # headers={"Content-Disposition": "attachment; filename=text.txt"},
#     #     headers={"Content-type": "text/plain"},
#     # )
#     return {"transcribe": 'done'}


def process_live_request(body):
    data = json.loads(body)

    wav_bytes = bytes(data['data'])
    uuid = data['uuid']
    end_of_sentence = data['end']

    if not buffer_dict[uuid]:
        buffer_dict[uuid] = io.BytesIO()
    buffer_dict[uuid].write(wav_bytes)
    buffer_dict[uuid].seek(0)
    bytes_data = buffer_dict[uuid].read()
    return bytes_data, uuid, end_of_sentence


# @app.post("/process_rtt_flush/")
@app.post("/transcribe_live/urdu")
async def process_rtt_endpoint(request: Request):
    print('transcribing urdu sentence...')
    body = await request.body()
    wav_bytes, uuid, end_of_sentence = process_live_request(body)
    transcription = transcribe_flutter(wav_bytes, 'urdu')
    if end_of_sentence:
        buffer_dict[uuid].close()
        del buffer_dict[uuid]
    print(transcription)
    return {"transcribe": base64.b64encode(transcription.encode()).decode()}


# def read_bytes_buffer(bytes):
#     global buffer
#
#     buffer_dict[uuid] = io.BytesIO()
#
#     buffer.write(bytes)
#
#     buffer.seek(0)
#     print(buffer.__sizeof__())
#     data = buffer.read()
#
#     return data


def transcribe_flutter(wav, lang):
    # wav = read_bytes_buffer(wav_bytes)
    if wav == b'':
        print('empty data')
        return ""
    # flutter
    # audio_array = AudioSegment.from_file(io.BytesIO(wav))
    wav_io = io.BytesIO()
    # todo, make sure below, if its the correct info from front end
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 1,2
        wav_file.setsampwidth(2)  # 2,3,4
        wav_file.setframerate(16000)  # 16000,22050, 48000
        # wav_file.writeframesraw(audio_array.raw_data)
        wav_file.writeframesraw(wav)
        # flutter
        # wav_file.writeframesraw(wav)
    res = wav_io.getvalue()

    data, samplerate = sf.read(io.BytesIO(res))

    if samplerate != 16000:
        data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)

    if lang == 'urdu':
        segments, info = modelUrdu.transcribe(data, vad_filter=False,
                                           language='ur',
                                           without_timestamps=True)  # ,vad_parameters=dict(min_silence_duration_ms=200), )
        transcription = ''
        for segment in segments:
            print("\nfrom whisper: [%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            transcription += segment.text

    else:
        segments, info = modelUrdu.transcribe(data, vad_filter=False,
                                           language='en',
                                           without_timestamps=True)  # ,vad_parameters=dict(min_silence_duration_ms=200), )
        transcription = ''
        for segment in segments:
            print("\nfrom whisper: [%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            transcription += segment.text

    return transcription
    # return transcription, transcription_lm



def convert_dia(intervals):
    merged_intervals = []
    c = 0
    merged_list = intervals
    max = len(merged_list) - 1
    fix_c = 0
    for i in range(max):
        print(i)
        print(max)
        if c == max: break
        if i < c: continue

        # if no overlapping, keep going
        while merged_list[c]['end'] <= merged_list[c + 1]['start']:

            if merged_intervals and merged_intervals[-1]['label'] == 'OVERLAP':
                merged_intervals.append({'start': merged_intervals[-1]['end'], 'end': merged_list[fix_c]['end'],
                                         'label': merged_list[c - 1]['label']})

                # merged_intervals.append({'start':merged_intervals[-1]['end'], 'end': merged_list[c-1]['end'], 'label': merged_list[c-1]['label']})
            else:
                merged_intervals.append(merged_list[c])
            fix_c = c
            c += 1
            if c == max:
                merged_intervals.append(merged_list[c])
                break
        # if merged_list[c]['end'] == merged_list[c + 1]['start']:
        #     merged_intervals.append({'start':merged_intervals[-1]['end'], 'end': merged_list[c+1]['start'], 'label': merged_list[fix_c]['label']})

        # if overlapping,
        if c == max: break
        if merged_list[c]['end'] <= merged_list[c + 1]['end']:
            # if c==max: break
            if merged_intervals and merged_intervals[-1]['label'] == 'OVERLAP':
                merged_intervals.append({'start': merged_intervals[-1]['end'], 'end': merged_list[c]['end'],
                                         'label': merged_list[c]['label']})
                c += 1
                if c == max:
                    merged_intervals.append(merged_list[c])
                    break
                # merged_intervals.append({'start':merged_list[c-1]['end'], 'end': merged_list[c]['end'], 'label':merged_list[c]['label']})

            else:

                merged_intervals.append({'start': merged_list[c]['start'], 'end': merged_list[c + 1]['start'],
                                         'label': merged_list[c]['label']})
                merged_intervals.append(
                    {'start': merged_list[c + 1]['start'], 'end': merged_list[c]['end'], 'label': 'OVERLAP'})
                c += 1
                if c == max:
                    merged_intervals.append({'start': merged_intervals[-1]['end'], 'end': merged_list[c]['end'],
                                             'label': merged_list[c]['label']})
                    break
            # fix_c=c



        else:
            fix_c = c
            while merged_list[fix_c]['end'] > merged_list[c + 1]['end']:
                if merged_intervals and merged_intervals[-1]['label'] == 'OVERLAP':
                    merged_intervals.append({'start': merged_intervals[-1]['end'], 'end': merged_list[c + 1]['start'],
                                             'label': merged_list[fix_c]['label']})
                else:
                    merged_intervals.append({'start': merged_list[c]['start'], 'end': merged_list[c + 1]['start'],
                                             'label': merged_list[fix_c]['label']})
                merged_intervals.append(
                    {'start': merged_list[c + 1]['start'], 'end': merged_list[c + 1]['end'], 'label': 'OVERLAP'})
                c += 1
                if c == max:
                    merged_intervals.append({'start': merged_intervals[-1]['end'], 'end': merged_list[fix_c]['end'],
                                             'label': merged_list[fix_c]['label']})
                    break

    return merged_intervals


def merge_list(dia):
    merge_list = []
    i = 0
    prev_speaker = dia[0]['label']
    start = dia[0]['start']
    first_time = True
    while i < len(dia):
        # start = dia[i]['start']
        if dia[i]['label'] != prev_speaker:
            merge_list.append({'start': start, 'end': dia[i - 1]['end'], 'label': dia[i - 1]['label']})
            first_time = True
            prev_speaker = dia[i]['label']

        if dia[i]['label'] == prev_speaker:
            if first_time:
                start = dia[i]['start']
                first_time = False
        i += 1
    return merge_list


# def generate(wav_bytes, lang):
#     print('file')
#     #todo empty file will raise issue here
#     a = pydub.AudioSegment.from_file(io.BytesIO(wav_bytes))
#
#     sampling_rate = 16000
#     if a.frame_rate != sampling_rate:
#         a = a.set_frame_rate(sampling_rate)
#     if a.channels != 1:
#         a = a.set_channels(1)
#
#     (get_speech_timestamps, _, read_audio, *_) = utils
#
#     wav_io = io.BytesIO()
#     with wave.open(wav_io, 'wb') as wav_file:
#         wav_file.setnchannels(a.channels)  # 1,2
#         wav_file.setsampwidth(a.sample_width)  # 2,3,4
#         wav_file.setframerate(a.frame_rate)  # 16000,22050, 48000
#         wav_file.writeframesraw(a.raw_data)
#     res = wav_io.getvalue()
#
#     # wav = read_audio(io.BytesIO(res), sampling_rate=sampling_rate)
#
#     # segments = pydub.silence.split_on_silence(a, 100, silence_thresh=-16)  # Adjust the threshold as needed
#     # time_stamp = pydub.silence.detect_nonsilent(a, 100, silence_thresh=-16)  # Adjust the threshold as needed
#
#     dia_results = diarization(io.BytesIO(res))
#
#
#
#     list_dia =[]
#
#
#     for x in dia_results._tracks.items():
#         list_dia.append({'start': int(x[0].start *16000), 'end': int(x[0].end *16000), 'label': [*x[1].values()][0]})
#
#
#     for x in list_dia:
#         print({'start': (x['start'] / 16000), 'end': (x['end'] / 16000), 'label': x['label']})
#
#     print("here")
#     # list_dia = merge_list(list_dia)
#     list_dia = convert_dia(list_dia)
#
#
#
#
#     z=[]
#     for x in dia_results._tracks.items():
#         z.append({'start': int(x[0].start )*16000, 'end': int(x[0].end )*16000, 'label': [*x[1].values()][0]})
#     s = []
#     for x in z:
#         #  {'start': 28046880, 'end': 28115424},
#         s.append({'start': (x['start'] / 16000), 'end': (x['end'] / 16000), 'label': x['label']})
#
#
#
#
#
#
#     i = 0
#     for dia in list_dia:
#
#         temp = a.get_sample_slice(dia['start'], dia['end'])
#         # temp.export(f'./temp/{i-1}.wav', format ='wav')
#         i+=1
#         wav_io = io.BytesIO()
#         with wave.open(wav_io, 'wb') as wav_file:
#             wav_file.setnchannels(temp.channels)  # 1,2
#             wav_file.setsampwidth(temp.sample_width)  # 2,3,4
#             wav_file.setframerate(temp.frame_rate)  # 16000,22050, 48000
#             wav_file.writeframesraw(temp.raw_data)
#         res = wav_io.getvalue()
#
#         data, samplerate = sf.read(io.BytesIO(res))
#
#         # speech_timestamps = get_speech_timestamps(wav, vad, sampling_rate=sampling_rate)
#
#         # for seg in speech_timestamps:
#         #     temp1 = a.get_sample_slice(seg['start'], seg['end'])
#         #
#         #     wav_io = io.BytesIO()
#         #     with wave.open(wav_io, 'wb') as wav_file:
#         #         wav_file.setnchannels(temp1.channels)  # 1,2
#         #         wav_file.setsampwidth(temp1.sample_width)  # 2,3,4
#         #         wav_file.setframerate(temp1.frame_rate)  # 16000,22050, 48000
#         #         wav_file.writeframesraw(temp1.raw_data)
#         #     res = wav_io.getvalue()
#         #
#         #     data, samplerate = sf.read(io.BytesIO(res))
#

#         # full_transcription = full_transcription + '\n' + milliseconds_to_time(seg) + transcription
#         full_transcription = milliseconds_to_time(dia) + f"-->{dia['label']} :   " + '    ' + translation + '\n'
#         print(full_transcription)
#         yield full_transcription

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    # uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info",ssl_certfile="cert.crt", ssl_keyfile="cert.key")
    # uvicorn.run("app:app")

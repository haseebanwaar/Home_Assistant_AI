
import base64
import datetime
import io
import json
import re
import time
import uuid
import wave

import cv2
import nest_asyncio
import numpy as np
import pydub
import soundfile as sf
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from line_profiler_pycharm import profile
from starlette.responses import StreamingResponse

from providers.asr.whisper import whisper_transcribe
from providers.local_openAI import client#, model_llm
from providers.tts.kokoro import kokoro_tts
from providers.animation.float.generate import InferenceAgent, InferenceOptions
nest_asyncio.apply()









app = FastAPI()
opt = InferenceOptions().parse()
opt.rank, opt.ngpus = 0, 1
video_anime = InferenceAgent(opt)

vad, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                            model='silero_vad',
                            # source='local',  # )
                            onnx=True)
(get_speech_timestamps, _, read_audio, *_) = utils
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat/audio")
async def live_chat(request: Request):
    video = False
    wav_bytes = await request.body()
    wav_bytes = wav_bytes.decode('utf-8')
    wav_bytes = json.loads(wav_bytes)
    wav_bytes_audio = wav_bytes['data']
    # wav_bytes_audio = wav_bytes['audio']
    wav_bytes_image = wav_bytes['image'] if 'image' in wav_bytes.keys() else None
    wav_bytes_video = wav_bytes['video'] if 'video' in wav_bytes.keys() else None

    return StreamingResponse(
        generate_flutter(wav_bytes_audio, wav_bytes_image, wav_bytes_video, video),
        media_type="text/plain",
        headers={"Content-type": "text/plain"},
    )

@app.post("/chat/video")
async def live_video(request: Request):
    video = True
    wav_bytes = await request.body()
    wav_bytes = wav_bytes.decode('utf-8')
    wav_bytes = json.loads(wav_bytes)
    wav_bytes_audio = wav_bytes['data']
    # wav_bytes_audio = wav_bytes['audio']
    wav_bytes_image = wav_bytes['image'] if 'image' in wav_bytes.keys() else None
    wav_bytes_video = wav_bytes['video'] if 'video' in wav_bytes.keys() else None

    return StreamingResponse(
        generate_flutter(wav_bytes_audio, wav_bytes_image, wav_bytes_video, video),
        media_type="text/plain",
        headers={"Content-type": "text/plain"},
    )


@profile
async def generate_flutter(wav_bytes_audio, wav_bytes_image, wav_bytes_video, video):
    tim = time.perf_counter()
    res = bytes(wav_bytes_audio)
    a = pydub.AudioSegment.from_raw(io.BytesIO(res), sample_width=2, frame_rate=16000, channels=1)

    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(a.channels)  # 1,2
        wav_file.setsampwidth(a.sample_width)  # 2,3,4
        wav_file.setframerate(a.frame_rate)  # 16000,22050, 48000
        wav_file.writeframesraw(a.raw_data)
    res = wav_io.getvalue()

    wav = read_audio(io.BytesIO(res), sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, vad, sampling_rate=16000,
                                              visualize_probs=True, threshold=0.4)
    # todo, changes these settings might help
    # speech_timestamps = get_speech_timestamps(wav, vad, sampling_rate=sampling_rate,
    #                                       visualize_probs=True, return_seconds=True, threshold=0.2, min_speech=0.1,
    #                                       min_silence=0.1)

    print(speech_timestamps)
    print(f'total segments: {len(speech_timestamps)}')
    full_transcription = ''
    for seg in speech_timestamps:
        temp = a.get_sample_slice(seg['start'], seg['end'])

        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(temp.channels)  # 1,2
            wav_file.setsampwidth(temp.sample_width)  # 2,3,4
            wav_file.setframerate(temp.frame_rate)  # 16000,22050, 48000
            wav_file.writeframesraw(temp.raw_data)
        res = wav_io.getvalue()

        data, samplerate = sf.read(io.BytesIO(res))
        full_transcription += whisper_transcribe(data)


    image = video_path = ''
    if wav_bytes_image:
        image_data = base64.b64decode(wav_bytes_image)
        image_array = np.frombuffer(image_data, np.uint8)  # Convert to NumPy array
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Decode into OpenCV format

    if wav_bytes_video:
        video_path = Image.open(io.BytesIO(base64.b64decode(wav_bytes_video)))

    sentences = re.split('([.?])',full_transcription)
    final_sentences = [sentences[i] + sentences[i + 1] for i in range(0, len(sentences) - 1, 2)]
    # final_sentences=['Hello, how are you. i was thinking we should go to Rome and have fun. '
    #                  'Do you have any other plans. are you good to go?ok i am heading out. talk to you later!']
    # final_sentences=['Hello, how are you?','i was thinking we should go to Rome and have fun',
    #                  'Do you have any other plans','are you good to go?','ok i am heading out','talk to you later']

    # for s in full_transcription:
    #     if not video:
    #         data = kokoro_tts(s)
    #     else:
    #         audio = kokoro_tts(s)
    #         wav_buffer = io.BytesIO(audio)
    #
    #         call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    #         data = video_anime.run_inference(
    #             f'./providers/temp_results/{call_time}.mp4',
    #             image,
    #             wav_buffer,
    #             a_cfg_scale=2.0,
    #             r_cfg_scale=1.0,
    #             e_cfg_scale=1.0,
    #             emo='neutral',
    #         )


    audio = kokoro_tts(full_transcription)
    wav_buffer = io.BytesIO(audio)

    call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    data = video_anime.run_inference(
        f'./providers/temp_results/{call_time}.mp4',
        image,
        wav_buffer,
        a_cfg_scale=2.0,
        r_cfg_scale=1.0,
        e_cfg_scale=1.0,
        emo='neutral',
    )

    yield f"Query: {full_transcription}\n".encode('utf-8')
    print(f'sent text of: {full_transcription}')
    data_base64 = base64.b64encode(data).decode('utf-8')
    yield f"Response: {data_base64}\n".encode('utf-8')
    print(f'video text of: {full_transcription}')

    print(time.perf_counter()-tim)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    # uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info",ssl_certfile="cert.crt", ssl_keyfile="cert.key")
    # uvicorn.run("app:app")







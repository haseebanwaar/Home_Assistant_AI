import base64
import datetime
import io
import json
import re
import asyncio
import time
import uuid
import wave

import cv2
import nest_asyncio
import numpy as np
import pydub
import requests
import soundfile as sf
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from line_profiler_pycharm import profile
from starlette.responses import StreamingResponse

from agents.event_bus import EventBus
from agents.event_extractor_agent import EventExtractorAgent
from agents.perception_agent import PerceptionAgent
from agents.talker_agent import TalkerAgent
# from providers.asr.parakeet import nemo_transcribe
from providers.local_openAI import client, get_model_name_vlm
# from providers.tts.Orpheus.orpheus import run_orpheus
from providers.tts.kokoro.kokoro_tts import run_kokoro
from vecttor_store.activity_logger import ActivityLogger
from sources.screen import RealtimeScreenCapture
nest_asyncio.apply()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name_vlm = ''


async def get_running_model():
    global model_name_vlm
    model_name_vlm = await get_model_name_vlm()
    print("Model:", model_name_vlm)


asyncio.run(get_running_model())

# In-memory store for the conversation history for a single user.
_chat_history = []


async def main():
    event_bus = EventBus()
    vlm = get_running_model()  # your single shared VLM
    qdrant = ActivityLogger()

    agents = [
        PerceptionAgent(vlm, event_bus, RealtimeScreenCapture, embed_func, qdrant),
        EventExtractorAgent(vlm, event_bus),
        FaceRecognitionAgent(vlm, event_bus, face_db),
        TalkerAgent(vlm, event_bus, run_kokoro)
    ]

    for agent in agents:
        asyncio.create_task(agent.run())

    await event_bus.run_forever()


@app.post("/chat/audio")
async def live_chat(request: Request):
    video = False
    wav_bytes = await request.body()
    wav_bytes = wav_bytes.decode('utf-8')
    wav_bytes = json.loads(wav_bytes)
    wav_bytes_audio = wav_bytes['data']

    wav_bytes_image = wav_bytes.get('image')
    wav_bytes_video = wav_bytes.get('video')
    clear_history = wav_bytes.get('clear_history', False)

    if clear_history:
        _chat_history.clear()
        print("Chat history cleared.")

    return StreamingResponse(
        generate_response(wav_bytes_audio, wav_bytes_image, wav_bytes_video, video, _chat_history),
        media_type="application/x-ndjson",
        headers={"Content-type": "text/plain"},
    )


@profile
async def generate_response(wav_bytes_audio, wav_bytes_image, wav_bytes_video, video, chat_history):
    res = bytes(wav_bytes_audio)
    a = pydub.AudioSegment.from_raw(io.BytesIO(res), sample_width=2, frame_rate=16000, channels=1)

    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(a.channels)  # 1,2
        wav_file.setsampwidth(a.sample_width)  # 2,3,4
        wav_file.setframerate(a.frame_rate)  # 16000,22050, 48000
        wav_file.writeframesraw(a.raw_data)
    res = wav_io.getvalue()

    tim = time.perf_counter()
    data, samplerate = sf.read(io.BytesIO(res))
    # full_transcription = nemo_transcribe(data)
    full_transcription = """"""
    print(f'ASR took: {time.perf_counter() - tim} seconds')

    user_content = [
        {"type": "text", "text": f'{full_transcription}'},
    ]
    if wav_bytes_image:
        user_content.insert(0, {
            "type": "image_url",
            "image_url": {
                'url': f'data:image/jpeg;base64,{wav_bytes_image}'
            },
        })

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    messages.extend(chat_history)
    messages.append({
        "role": "user",
        "content": user_content,
    })

    chat_response = await client.chat.completions.create(
        model=model_name_vlm,
        messages=messages,
        stream=True,  # Enable streaming from the VLM
    )

    # 1. Send the user's transcribed query text to the client
    query_payload = {
        "type": "query",
        "text": full_transcription
    }
    yield (json.dumps(query_payload) + "\n").encode('utf-8')
    print(f"Sent query text: {full_transcription}")

    # 2. Stream the VLM response and TTS audio
    full_assistant_response = ""
    async for chunk, text_chunk in stream_vlm_and_audio(chat_response):
        yield chunk
        if text_chunk:
            full_assistant_response += text_chunk

    # 3. Update the stateful history with the user query and full assistant response
    chat_history.append({"role": "user", "content": user_content})
    chat_history.append({"role": "assistant", "content": full_assistant_response})


@profile
async def stream_vlm_and_audio(chat_response_stream):
    """
    Streams VLM text sentence by sentence and generated audio in batches.
    Yields the audio/text payload for the client and the raw text for history.
    """
    full_sentence = ""
    sentence_buffer_for_audio = []
    audio_tasks = []  # (task, sentences_text)
    sentence_count = 0  # Track number of sentences processed

    async def generate_audio_task(text):
        """Helper to run TTS in thread pool and return audio payload"""
        audio_bytes = await asyncio.to_thread(run_kokoro, text)
        # audio_bytes = await asyncio.to_thread(run_orpheus, text)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {"type": "audio", "data": audio_base64}

    def split_into_chunks(text, max_words=30):
        """Split long text into word chunks if no delimiter found."""
        words = text.split()
        for i in range(0, len(words), max_words):
            yield " ".join(words[i:i + max_words])

    async for chunk in chat_response_stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_sentence += delta

            # Check if we have a complete sentence (delimiter found or 30 words reached)
            if any(punct in full_sentence for punct in ".!?") or len(full_sentence.split()) >= 30:
                # Split sentence into smaller pieces if needed
                chunks = list(split_into_chunks(full_sentence.strip()))

                for sentence_to_send in chunks:
                    # ---- Send VLM text immediately ----
                    vlm_text_payload = {"type": "vlm_text", "text": sentence_to_send}
                    yield (json.dumps(vlm_text_payload) + "\n").encode("utf-8"), sentence_to_send
                    print(f"Sent VLM text: '{sentence_to_send}'")

                    # ---- Add sentence to the audio batching buffer ----
                    sentence_buffer_for_audio.append(sentence_to_send)
                    sentence_count += 1

                    # ---- Determine batch size ----
                    AUDIO_BATCH_SIZE = 1 if sentence_count <= 3 else 3

                    # ---- Create a background audio task when the batch is full ----
                    if len(sentence_buffer_for_audio) >= AUDIO_BATCH_SIZE:
                        text_to_speak = " ".join(sentence_buffer_for_audio)
                        task = asyncio.create_task(generate_audio_task(text_to_speak))
                        audio_tasks.append((task, text_to_speak))
                        sentence_buffer_for_audio.clear()

                full_sentence = ""

        # ---- Yield finished audio tasks in order ----
        while audio_tasks and audio_tasks[0][0].done():
            task, spoken_text = audio_tasks.pop(0)
            audio_payload = await task
            yield (json.dumps(audio_payload) + "\n").encode("utf-8"), None
            print(f"Sent batched audio for: '{spoken_text}'")

    # ---- Flush remaining sentences ----
    if sentence_buffer_for_audio:
        text_to_speak = " ".join(sentence_buffer_for_audio)
        task = asyncio.create_task(generate_audio_task(text_to_speak))
        audio_tasks.append((task, text_to_speak))

    # ---- Flush remaining audio tasks in order ----
    for task, spoken_text in audio_tasks:
        audio_payload = await task
        yield (json.dumps(audio_payload) + "\n").encode("utf-8"), None
        print(f"Sent final batched audio for: '{spoken_text}'")


if __name__ == "__main__":
    asyncio.run(main())
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

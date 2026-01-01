import base64
import datetime
import io
import json
import re
import asyncio
import time
import wave

import cv2
import nest_asyncio
import numpy as np
import pydub
import requests
import soundfile as sf
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from line_profiler_pycharm import profile
from qdrant_client import QdrantClient
from lmdeploy.vl.utils import encode_image_base64
from starlette.responses import StreamingResponse

from agents.event_bus import EventBus
from agents.event_extractor_agent import EventExtractorAgent
from agents.perception_agent import PerceptionAgent
from agents.talker_agent import TalkerAgent
from providers.asr.parakeet import nemo_transcribe
# from providers.asr.parakeet import nemo_transcribe
from providers.local_openAI import client, get_model_name_vlm
# from providers.tts.Orpheus.orpheus import run_orpheus
from providers.tts.kokoro.kokoro_tts import run_kokoro
from vector_store.activity_logger import ActivityLogger
from sources.screen import RealtimeScreenCapture
from sources.rtsp import RealtimeCameraStream
from vector_store.rag.activity_retriever import ActivityRetriever


nest_asyncio.apply()

app = FastAPI(title="Home Assistant AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === GLOBALS ===
_chat_history = []
event_bus = EventBus()
vlm_model = None
talker_agent = None
perception_agent = None
screen_stream = None

# Create a single, shared Qdrant client instance
qdrant_client = QdrantClient(path='./qdrant_db')

past_memory = ActivityRetriever(client=qdrant_client)
activity_logger = ActivityLogger(client=qdrant_client)

# === STARTUP ===
@app.on_event("startup")
async def startup_event():
    global vlm_model, talker_agent, perception_agent, screen_stream, activity_logger

    print("Loading model...")
    vlm_model = await get_model_name_vlm()
    print(f"✅ Model loaded: {vlm_model}")
    screen_stream = RealtimeScreenCapture(
        video_source="",  # Not used
        model_name_vlm=vlm_model,
        window_size=60,

        monitor_index=2,  # Capture the primary monitor
        activity_logger=activity_logger  # we pass activity logger here
        # target_resolution=(1720, 720), # Reduce resolution to 1280x720
    )
    talker_agent = TalkerAgent(vlm_model, event_bus, run_kokoro)
    # perception_agent = PerceptionAgent(vlm_model, event_bus, run_kokoro)
    # asyncio.create_task(talker_agent.run())
    asyncio.create_task(event_bus.run_forever())
    print("✅ TalkerAgent and EventBus started.")


# === API ===
@app.post("/chat/audio")
async def live_chat(request: Request):
    body = await request.body()
    data = json.loads(body.decode("utf-8"))

    wav_bytes_audio = data["data"]
    wav_bytes_image = data.get("image")
    wav_bytes_video = data.get("video")  # do i really need it?
    clear_history = data.get("clear_history", False)
    concise = data.get("talking", False)
    context = data.get("context")
    live = data.get("live")
    memory = data.get("memory")

    if clear_history:
        _chat_history.clear()
        print("Chat history cleared.")

    return StreamingResponse(
        generate_response(wav_bytes_audio, wav_bytes_image, _chat_history, concise, context,live, memory),
        media_type="application/x-ndjson"
    )

MEMORY_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_memory",
            "description": "Search user history from vector store",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": { "type": "string" ,                        "description": "The topic to look for. String value for semantic search in vector db."
},
                    "time_value": { "type": "number" ,                        "description": "The numerical value for the time range (e.g., 2.5, 10, 1)."
},
                    "time_unit": {
                        "type": "string",
                        "enum": ["minutes", "hours", "days", "weeks", "months"],
                        "description": "The unit of time to look back."

                    },
                    "date": {
                        "type": "string",
                        "description": "Absolute date in YYYY-MM-DD"
                    }
                }
            }
        }
    }
]

# === RESPONSE GENERATION ===
async def generate_response(wav_bytes_audio, wav_bytes_image, chat_history, concise, context, live, memory):
    """Handle incoming audio and generate streamed text + TTS output."""
    # Convert audio bytes to WAV
    res = bytes(wav_bytes_audio)
    audio_seg = pydub.AudioSegment.from_raw(io.BytesIO(res), sample_width=2, frame_rate=16000, channels=1)
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(audio_seg.channels)
        wav_file.setsampwidth(audio_seg.sample_width)
        wav_file.setframerate(audio_seg.frame_rate)
        wav_file.writeframesraw(audio_seg.raw_data)
    res = wav_io.getvalue()

    tim = time.perf_counter()
    data, samplerate = sf.read(io.BytesIO(res))
    transcription = nemo_transcribe(data)
    # transcription = "what is happening ?"
    print(f'ASR took: {time.perf_counter() - tim} seconds')

    # Prepare prompt
    user_content = [{"type": "text", "text": transcription}]
    if wav_bytes_image and not live:
        user_content.insert(0, {
            "type": "image_url",
            "image_url": {
                'url': f'data:image/jpeg;base64,{wav_bytes_image}'
            },
        })

    if live :
        # lets suppose i m watching movie, and every minute context of the movie is saved by Realtimescreen capture
        # in vector db with timestamps. now if user asks after watching movie 15 mins, 'what the story so far?'
        # so as live is True, ill also transcribe remaining frames in queue so i am live, not <min in past.
        if context == 'screen':
            for img in screen_stream.frame_buffer:
                user_content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 9, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})
        elif context == 'camera':
            for img in RealtimeCameraStream.frame_buffer:
                user_content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 9, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})

    # 2. Initialize messages with the base system prompt
    if not concise:
        system_prompt_content = "You are a helpful assistant."
    else:
        system_prompt_content = """You are a conversational AI designed for a real-time Speech-to-Speech (S2S) system. Your primary function is to engage in natural, fluid conversation.

    Follow these critical rules:
    1.  **Be Concise:** Keep your responses short, typically one or two sentences. Avoid long paragraphs at all costs.
    2.  **Sound Natural:** Speak like a real person. Use contractions (e.g., "it's," "don't," "you're") and a friendly, conversational tone.
    3.  **TTS-Friendly:** Your responses will be spoken aloud by a Text-to-Speech (TTS) engine. Use simple sentence structures and common vocabulary that are easy to pronounce and sound natural when spoken.
    4.  **No Formatting:** Do not use lists, bullet points, markdown, or any text formatting. Your output is for voice only.

    Your goal is to keep the conversation moving, not to provide exhaustive, written-out answers.
    """

    messages = [{"role": "system", "content": system_prompt_content}]

    # 3. Add past memory if available
    past_activities = None
    if memory:
        # so continue the prev discussion of watching movie, live is True so fresh description of frames is also added
        # but user asked what has happened uptil now, that 15 mins of movie, ill need to fetch last 15 mins of
        # db content with flag 'screen' or i should also introduce sub flag that in addition differentiating bw screen
        # and camera also helps put boundries onuser task, like user sets it to moviegame, coding, browsing manually from UI
        # so how it all goes?

        memmsg=[{"role": "system",
                 "content": "you are provided with callable functions, based on user query decide with what parameters you will call it with"}]
        memmsg.append({"role": "user", "content": transcription})
        chat_response = await client.chat.completions.create(
            model=vlm_model,
            messages=memmsg,
            tools=MEMORY_TOOL_SCHEMA,
            tool_choice="auto",
        )

        choice = chat_response.choices[0]
        tool_calls = choice.message.tool_calls
        tool_call = tool_calls[0]

        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)

        if fn_name == "retrieve_memory":
            result = past_memory.retrieve_memory(
                search_query=fn_args.get("search_query"),
                time_value=fn_args.get("time_value"),
                time_unit=fn_args.get("time_unit"),
            )


        # past_activities = past_memory.retrieve_memory(transcription, context,screen_stream.current_minute_apps[-1])
            print(f"function args: {fn_args}")
            if result:
                if isinstance(result, list):
                    past_activities_text = "\n".join(result)
                else:
                    past_activities_text = str(result)

                #todo, count tokens here and summerize if memory was too long with another vlm call
                messages.append({"role": "user", "content": f"Here is some relevant past memory for context: {past_activities_text}"})

    # 4. Extend with chat history
    if chat_history:
        messages.extend(chat_history)

    # 5. Append the current user message
    messages.append({"role": "user", "content": user_content})

    chat_response = await client.chat.completions.create(
    model=vlm_model,
    messages=messages,
    stream=True
    )

    yield json.dumps({"type": "query", "text": transcription}) + "\n"

    # 2. Stream the VLM response and TTS audio
    full_assistant_response = ""
    async for chunk, text_chunk in stream_vlm_and_audio(chat_response):
        yield chunk
        if text_chunk:
            full_assistant_response += text_chunk

    # 3. Update the stateful history with the user query and full assistant response
    chat_history.append({"role": "user", "content": user_content})
    chat_history.append({"role": "assistant", "content": full_assistant_response})


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
    # asyncio.run(main())
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# Home Assistant AI
<img width="1024" height="490" alt="image" src="https://github.com/user-attachments/assets/dd2441e4-2e12-40cc-80eb-a4e9d3e181ff" />

**Description:**
Towards Agents-based Home and Personal Assistant AI (LLM, VLM, Omni)

## What does this project do?
Home Assistant AI is a multimodal personal assistant that listens to your speech, understands it, and speaks back. It combines:
- Voice Activity Detection (Silero VAD) to split your audio into speech segments.
- Automatic Speech Recognition (Whisper) to transcribe each spoken segment.
- An LLM/VLM accessed via an OpenAI-compatible local server to understand context and optionally reason over images/video.
- Text-to-Speech (multiple backends like Kokoro, Orpheus, Chatterbox, Kitten) to generate a natural-sounding voice response.
- Optional visual response generation hooks for talking-head video (experimental, commented in code).

In short: you send audio (and optionally an image/video), it transcribes, reasons, and returns spoken (or text) responses through a simple FastAPI server.

## Overview
Home Assistant AI is an initiative to develop a next-generation, agents-based system for home and personal assistant functionality. This project leverages the power of Large Language Models (LLMs), Vision-Language Models (VLMs), and omnidirectional capabilities to create a versatile, intelligent, and adaptive assistant.

## Quick start
- Install dependencies (PyTorch, FastAPI, uvicorn, pydub, soundfile, OpenAI client, etc.).
- Run the API server: `python app.py` (it launches uvicorn at http://0.0.0.0:8000).
- POST to one of the endpoints with base64-encoded payloads:
  - POST /chat/audio — body: `{ "data": <base64-audio>, "image"?: <base64-image>, "video"?: <base64-video> }`
  - POST /chat/video — same body; enables experimental video mode.
- The server performs VAD + ASR, then TTS, and streams a textual summary. Audio bytes from TTS are produced internally (see providers/tts/*), and can be adapted to your client needs.

## Goals
1. **Intelligent Agents**: Develop modular agents for specific tasks (e.g., scheduling, reminders, home automation).
2. **Multimodal Interactions**: Utilize both language and vision capabilities for rich, context-aware interactions.
3. **Personalization**: Adapt to individual user preferences and habits over time.
4. **Seamless Integration**: Work across various platforms and devices.

## Components
### ASR
- Whisper-based transcription via providers/asr/whisper.py.

### LLM/VLM
- OpenAI-compatible client (providers/local_openAI.py) targeting a locally served model (e.g., via llama.cpp, vLLM, LMDeploy). Adjust base_url/model per your setup.

### TTS
| Model | Realtime Factor | VRAM (GB) | Quality | Extra | Zero-shot Voice Cloning |Sample|
|---|---|---|---|---|---|---|
| VibeVoice 1.5B (bfloat16) | 1.65 | 6 | high | Expressive | yes |---|
| Kokoro | 0.02 | 2.5 | low | Bit robotic | - |---|
| Orpheus 0.1 ft (Q2_k gguf) | 0.63 | 2.3 | high | Expressive, emotions | - |---|
| Chatterbox | 1 | 5.2 | high | Plain | yes |---|
| Kitten TTS | 0.44 | 0.8 | low | Noisy, robotic | - |---|

## API endpoints
- POST /chat/audio — streams back the recognized query and runs TTS; accepts optional image/video for multimodal context.
- POST /chat/video — same as above but toggles experimental video response path (currently commented in code).

## Notes
- Silero VAD parameters (thresholds) are configurable in app.py.
- Example scripts live in examples/ for agents and VLM usage.
- Some TTS and video features are experimental; see providers/tts and comments in app.py for guidance.

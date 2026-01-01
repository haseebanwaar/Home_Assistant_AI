# Home Assistant AI
<img width="1024" height="490" alt="image" src="https://github.com/user-attachments/assets/dd2441e4-2e12-40cc-80eb-a4e9d3e181ff" />

Home/Personal Companion AI
Towards Agents-based Home and Personal Assistant AI (VLM)

📖 Overview
Home/Personal Companion AI is a next-generation, local-first multimodal assistant designed to perceive, reason, and interact with you in real-time. Unlike traditional voice assistants, this project bridges the gap between passive listening and active perception by integrating continuous screen and camera streams with long-term memory.

It leverages a Core  Vision-Language Models (VLM) for perception and reasoning. specifically we use qwen3-VL 8B hosted locally on vllm, thanks to 3d pooling, it can process 1 min video in 10 seconds on RTX 3090 and its unusually large context window and community support.  

🚀 Key Concepts & Architecture
The system operates on a modular pipeline designed for local deployment:

1. Multimodal Perception Streams
Voice Input: Captures audio commands via Parakeet ASR for english with very high RTF, or multilingual Whisper model. 

Screen Stream: A continuous feed of the user's desktop environment, allowing the AI to see active windows, read text on screen, and assist with workflows. detects change in on screen and sends it to VLM every minute, all the day and saves context in vector store.

Camera Stream: A live video feed providing real-world context, enabling the AI to "see" the user and their physical surroundings. detects change in all of your ip cameras and it stores interactions happening all the around your house in vector store 

2. The Core Processing Unit & Memory
The brain of the companion is a sophisticated orchestration of models:

Memory Retriever (RAG): Before answering, the system queries a Vector Store (Long-term Memory) using an Embedding Model.

Reranker: Retrieved memory chunks are re-ranked to ensure only the most contextually relevant historical data is fed to the LLM.

Reasoning and perception Engine (VLM): The core logic fuses current visual context (Screen/Camera), audio transcripts, and retrieved long-term memories to generate a response. 

3. Agent Spawner
The Core Unit is not just a chatbot; it acts as a dispatcher. Based on the complexity of the request, it can use an Agent Spawner to initialize specialized sub-agents (e.g., a "Coding Agent," "Calendar Agent," or "Search Agent") to execute multi-step tasks autonomously.

Talking Portrait (optional, computational heavy): The final response is delivered via high-quality TTS, which drives a visual "Talking Portrait" avatar for a face-to-face interaction experience.


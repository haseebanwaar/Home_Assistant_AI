# Home Assistant AI
<img width="1024" height="490" alt="image" src="https://github.com/user-attachments/assets/dd2441e4-2e12-40cc-80eb-a4e9d3e181ff" />

🤖 Home/Personal Companion AI
Towards Agents-based Home and Personal Assistant AI (VLM)
A next-generation, local-first multimodal assistant designed to perceive, reason, and interact in real-time.

📖 Overview
Home/Personal Companion AI bridges the gap between passive listening and active perception. Unlike traditional assistants, it integrates continuous Screen and Camera streams with Long-term Memory to create a companion that truly understands your context.

🧠 The Core Engine
At the heart of the system is the Core Vision-Language Model (VLM) for perception and reasoning.

Model: Qwen3-VL 8B

Hosting: Local vLLM instance.

Performance: Thanks to 3D pooling, it processes 1 minute of video in just 10 seconds on an RTX 3090.

Features: Leverages an unusually large context window and robust community support.

🚀 Key Concepts & Architecture
The system operates on a modular pipeline optimized for local deployment:

1. Multimodal Perception Streams 👁️
🎙️ Voice Input Captures audio commands using Parakeet ASR (English, very high Real-Time Factor) or the multilingual Whisper model.

🖥️ Screen Stream A continuous feed of your desktop environment. It detects on-screen changes, captures context every minute, all day, and saves it to the vector store. The AI sees active windows, reads text, and assists with workflows.

📹 Camera Stream Connects to your IP cameras to provide real-world context. It detects changes and interactions happening all around your house, storing physical-world events in the vector store.

2. The Core Processing Unit & Memory 💾
The "brain" is a sophisticated orchestration of models:

Memory Retriever (RAG): Before answering, the system queries the Vector Store (Long-term Memory) using an Embedding Model.

Reranker: Retrieved memory chunks are prioritized to ensure only the most contextually relevant history is fed to the model.

Reasoning & Perception Engine (VLM): The core logic fuses current visual context (Screen/Camera), audio transcripts, and retrieved memories to generate an informed response.

3. Agent Spawner 🛠️
The Core Unit acts as a Dispatcher, not just a chatbot.

Based on request complexity, it triggers the Agent Spawner.

Initializes specialized sub-agents (e.g., 🧑‍💻 Coding Agent, 📅 Calendar Agent, 🔍 Search Agent) to execute multi-step tasks autonomously.

4. Interactive Output 🗣️
Talking Portrait (Optional, Compute Heavy) The final response is delivered via high-quality TTS driving a visual "Talking Portrait" avatar, creating a genuine face-to-face interaction experience.

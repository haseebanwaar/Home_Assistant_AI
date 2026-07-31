# Home Assistant AI
<img width="1024" height="484" alt="image" src="https://github.com/user-attachments/assets/7bc0c609-4f9d-4cfe-bc20-6ec884d32aae" />

# 🤖 Home/Personal Companion AI
### Towards Agents-based Home and Personal Assistant AI (VLM)

> **A privacy and local-first cognitive companion designed to perceive, reason, and interact in real-time.**

---

## 📖 Overview

**Home/Personal Companion AI** bridges the gap between *passive listening* and *active perception*. Unlike traditional assistants, this project integrates continuous **Screen** and **Camera** streams with **Long-term Memory**, creating a companion that truly understands your context.

### 🧠 The Core Engine
At the heart of the system is the **Core Vision-Language Model (VLM)** for perception and reasoning.
- **Model:** Qwen3-VL 8B
- **Hosting:** Local `vLLM` instance.
- **Performance:** Thanks to 3D pooling, it processes **1 minute of video in just 10 seconds** on an RTX 3090.
- **Features:** Leverages an unusually large context window and robust community support.

---

## 🚀 Key Concepts & Architecture

The system operates on a modular pipeline designed for local deployment:

### 1. Multimodal Perception Streams 👁️
* **🎙️ Voice Input**
  Captures audio commands using **Parakeet ASR** (English, very high Real-Time Factor) or the multilingual **Whisper** model.

* **🖥️ Screen Stream**
  A continuous feed of your desktop environment. It detects on-screen changes, sends context to the VLM **every minute, all day**, and saves it to the vector store. The AI sees active windows, reads text, and assists with workflows.

* **📹 Camera Stream**
  Connects to your IP cameras to provide real-world context. It detects changes and interactions happening **all around your house**, storing physical-world events in the vector store.

### 2. The Core Processing Unit & Memory 💾
The "brain" is a sophisticated orchestration of models:
* **Memory Retriever (RAG):** Before answering, the system queries a **Vector Store** (Long-term Memory) using an Embedding Model.
* **Reranker:** Retrieved memory chunks are re-ranked to ensure only the most *contextually relevant* history is fed to the LLM.
* **Reasoning & Perception Engine (VLM):** The core logic fuses current visual context (Screen/Camera), audio transcripts, and retrieved long-term memories to generate a response.

### 3. Agent Spawner (Todo) 🛠️
The Core Unit acts as a **Dispatcher**, not just a chatbot. uses autogen for creating new agents.
* Based on request complexity, it triggers the **Agent Spawner**.
* Initializes specialized sub-agents (e.g., 🧑‍💻 *Coding Agent*, 📅 *Calendar Agent*, 🔍 *Search Agent*) to execute multi-step tasks autonomously.

### 4. Interactive Output 🗣️
* **Talking Portrait** *(Optional, Compute Heavy)*
  The final response is delivered via high-quality TTS driving a visual "Talking Portrait" avatar, creating a genuine face-to-face interaction experience.

---

### Agent orchestration

Every scheduled agent runs from one loop (`agents/orchestrator.py`) instead of
its own sleeper: clip retention, the nightly report, the next-day planner
lifecycle, memory consolidation, and a daily check-in per personal agent. Each
job declares its schedule, its priority, and whether it addresses the user.

Everything that speaks unprompted — scheduled check-ins and the proactive
narrator alike — claims from one shared `DeliveryBudget`, so two agents cannot
arrive in the same minute. A job denied a slot is deferred and delivers when the
user is no longer being interrupted; a job that claims a slot and then decides
it has nothing to say hands it back.

```dotenv
ORCHESTRATOR_TICK_SECONDS=30
AGENT_DELIVERY_GAP_SECONDS=300     # minimum gap between unprompted messages
AGENT_DELIVERIES_PER_HOUR=6
AGENT_CHECKINS_ENABLED=true
AGENT_CHECKIN_SCHEDULE=            # agent:wisdom=08:00,agent:roaster= (blank disables one)
AGENT_CHECKIN_TIMEOUT_SECONDS=240
```

`GET /orchestrator/status` reports every job's schedule, next due time, run,
failure and deferral counts, last error, and the current budget.
`POST /orchestrator/jobs/{job_id}/run` triggers one immediately — the schedule
is bypassed, the delivery budget is not.

### Long-term memory

The episodic graph (`Day → Session → Event`, plus `Entity`/`Claim`) never
compressed, so a question about last quarter meant scanning every minute ever
captured. `memory/consolidation.py` adds the coarse tier on top of it:

* **`Rollup`** nodes for each day, ISO week and calendar month, holding the
  period's metrics, highlights, entities and projects — plus the narrative the
  Coach wrote for it, so the nightly report becomes durable memory instead of a
  chat message no query can reach. `SUMMARIZES` links a rollup to its days and
  `ROLLS_UP_INTO` links each tier to the next, so a coarse answer can always be
  expanded back down to the events it came from.
* **`Project`** nodes promoted out of the `project_id` string on sessions, each
  with a lifespan and an active/dormant status recomputed every pass.
* **`Goal`** nodes promoted out of focus-session goals, linked to the sessions
  that pursued them.
* **Decay** — quarantined entities that never earned corroboration, and claims
  left without evidence, are pruned once they are old enough to be noise.
  Entities the user merged or renamed into are exempt at any age.

The pass runs nightly and is idempotent: rollups are recomputed from the base
events, which stay the source of truth.

```dotenv
MEMORY_CONSOLIDATION_ENABLED=true
MEMORY_CONSOLIDATION_AT=03:15
MEMORY_QUARANTINE_DAYS=45
PROJECT_DORMANT_DAYS=21
```

`GET /memory/long-term` returns the coarse tier — a fixed-size read whatever the
history length. `POST /memory/consolidate` runs the pass now, or backfills
history with `?start_date=&end_date=`. Agent rooms reach the same tier through
the `graph_long_term_memory` tool.

### Optional room agent runtime

The existing `FastAPI -> AsyncOpenAI -> Qwen` path remains the default. An
optional conversation manager can route explicitly selected rooms through
PydanticAI, using the same `AsyncOpenAI` client, model name, and
OpenAI-compatible endpoint. PydanticAI owns the multi-step model/tool loop for
those rooms and exposes MCP tools from a standard multi-server configuration.

Install the optional layer with:

```powershell
pip install -r requirements-agent.txt
```

The MCP extra pulls FastMCP, which tracks the newest Starlette. Starlette 1.x
drops the router arguments FastAPI 0.118 passes, so `requirements-agent.txt`
pins Starlette to a compatible range — installing it unpinned breaks `app.py`
at import time.

Enable the runtime; each room then persists Chat/Agent mode and its own toolset
allowlist in Room settings. The ID/kind variables remain a fallback for rooms
created before those settings existed:

```dotenv
AGENT_RUNTIME_ENABLED=true
AGENT_ROOM_IDS=
MCP_CONFIG_PATH=./mcp.config.json
AGENT_WORKSPACE_ROOT=C:/d/agent_workspaces
RESEARCH_AGENT_REQUEST_LIMIT=32
RESEARCH_AGENT_TOOL_CALLS_LIMIT=64
```

#### Tools

Agent rooms see two kinds of tool:

* **MCP servers** — everything external. Copy `mcp.config.example.json` to
  `mcp.config.json` and add filesystem, Git, GitHub, Playwright, or any other
  server under `mcpServers`. URL entries use Streamable HTTP (or SSE when the
  URL ends in `/sse`); command/args entries use stdio, and `${VAR}` /
  `${VAR:-default}` are expanded from the environment. The file is trusted
  configuration because stdio entries launch local processes. The shipped
  filesystem and headless Playwright entries self-start over stdio on Windows.
  `MCP_INIT_TIMEOUT_SECONDS` gives `npx` enough time for its MCP handshake.
  `${AGENT_WORKSPACE}` is resolved separately for every room at run time. In
  Room settings, leave **Agent workspace folder** blank for an automatic private
  folder, enter a relative name under `AGENT_WORKSPACE_ROOT`, or enter an
  absolute path. The built-in Research agent defaults to `research/` and creates
  `papers/`, `downloads/`, `protocol/`, `screening/`, `extraction/`, and
  `reports/`. The Playwright MCP output directory is also placed under that
  workspace so downloaded papers remain available to its filesystem tools.
* **The activity graph** (`agents/graph_tools.py`) — the one native toolset,
  because this application's own memory has no MCP server. It exposes
  `graph_search_memory`, `graph_recent_activity`, `graph_event_detail`,
  `graph_day_summary`, `graph_long_term_memory` and `graph_room_overview`. All
  are read-only and trim storage columns and long text before the results reach
  the model. `graph_long_term_memory` answers questions spanning weeks or months
  from the stored period summaries rather than from raw events.

MCP sessions open once per agent run rather than once per tool call. On Windows
this needs a subprocess-capable event loop, which is what uvicorn installs.

#### Responses

Direct room chat responses retain their existing shape. Agent responses add
`execution: agent` plus the PydanticAI run IDs and the tool call/output trace.
The built-in Research room defaults to Agent mode with graph, browser MCP, and
filesystem MCP access, and carries a reproducible SLR/PRISMA workflow prompt.
Research uses its own larger request/tool budget because a review may require
many search, download, read, and write turns. Agent chat uses the NDJSON endpoint
to show live activity: selected toolsets, tool calls with redacted argument
summaries, tool completion, drafting, and periodic still-working heartbeats.
Each Agent room can override **Model request limit** and **Tool call limit** in
Room settings; blank values inherit the normal or Research defaults.
Streaming room chat retains NDJSON; after the tool loop completes it emits the
final text and tool trace. `GET /agent-runtime/status` reports routing, limits,
native toolset count, configured MCP server names, and the selectable room tool
catalog without connecting to any server.

#### Structured outputs

Calls that need data rather than prose pass a Pydantic model as `output_type`
(see `agents/schemas.py`). Both paths honour it: PydanticAI validates and
retries on the agent path, and the direct path validates the same model after
extracting JSON from the reply. The next-day planner uses `PlanProposal` and
`PlanEvaluation` this way instead of scraping dictionaries out of free text.

One-shot extraction over evidence the caller already gathered passes
`allow_agent=False` to stay on the direct path even inside an agent room —
tools cannot improve the answer there and a stray tool loop can exhaust the
run's budget.

#### Tests

`tests/test_agent_runtime.py`, `tests/test_graph_tools.py` and
`tests/test_mcp_config.py` cover routing, toolset composition, config loading
and both structured-output paths. `tests/test_orchestrator.py` drives the
scheduler, the delivery budget and the shared context from a fake clock, and
`tests/test_consolidation.py` covers the long-term tier against a fake store —
neither needs Neo4j, an MCP server or the VLM. The live MCP tests spawn the filesystem and
Playwright servers over stdio and verify the Research room dependencies; they
are opt-in:

```powershell
$env:MCP_LIVE_TEST=1; python -m pytest tests/test_mcp_config.py
```

### Goals 
* **Productivity Enhancer**
  It enhances your productivty while working by helping you cambat distractions. it can watch videos along side you helping you grasp difficult concepts.

* **Proactivity**
  Proactive initiative is enabled by default across desktop, mobile, and camera
  observations. The model combines live context, graph-linked project/entity
  memory, personal preferences, active focus goals, and past reactions, then
  adapts its tone and varies its opening while deciding how to contribute.
  Cooldowns and duplicate checks pace delivery without prescribing which ideas
  the model may consider.

* **Evidence you can watch and question**
  Anything the assistant raises unprompted — a proactive nudge, an important or
  critical alert — carries the footage it was made from: a low-resolution,
  sped-up clip of the exact capture window (`GET /clips/<id>`, playable inline
  in the app). Those clips also answer follow-up questions from the video
  itself (`POST /clips/<id>/ask` → "was he carrying anything?"), so a claim
  about something you did not see can be checked instead of just believed.
  Clips a notification, nudge or question referenced are kept for days; every
  other clip is deleted within the hour (`sources/clips.py`).

* **On-demand reflection**
  `Alt+Shift+W` reflects on the current screen or camera context.
  `Alt+Shift+S` first asks whether to attach the PC screen, mobile screen,
  mobile camera, or a particular home camera. Both bindings are configurable
  in Settings and registered globally on desktop while the app is running;
  equivalent touch controls are shown in the Android UI.

  Guided reflection adds nine configurable prompt shortcuts:

  * `Alt+Shift+1` — Do you agree?
  * `Alt+Shift+2` — Critically analyze
  * `Alt+Shift+3` — Explain this simply
  * `Alt+Shift+4` — Distill key points
  * `Alt+Shift+5` — Explain this code
  * `Alt+Shift+6` — Review this code
  * `Alt+Shift+7` — Am I doing this okay?
  * `Alt+Shift+8` — What should I do next?
  * `Alt+Shift+9` — Challenge this

  The same actions are available from the in-app prompt palette for touch use
  or when remembering a direct binding would interrupt the task.

* **Lifelong Learning**
  The agent evolves alongside you. By continuously consolidating daily logs and visual context into its vector store, it builds a permanent, growing knowledge base. It remembers your preferences, projects, and history, ensuring that its personalization deepens over months and years of usage.


## ⚡ Quick Stats

| Component | Technology | Speed/Capability |
| :--- | :--- | :--- |
| **VLM Engine** | Qwen3-VL 8B (vLLM) | 1 min video → 10s processing (RTX 3090) |
| **ASR** | Parakeet / Whisper | Ultra-low latency / Multilingual |
| **Memory** | Vector Store + Reranker | Full-day context retention |

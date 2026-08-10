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
- **Model:** Qwen3.6 35B-A3B with vision projection
- **Hosting:** Local `llama.cpp` server on port 8888.
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

  Each camera also keeps a **persistent scene** rather than a pile of independent clips: every tracked thing (the gate, a parked car) holds a state with the moment it entered it, so a clip reads *"the gate has not opened since 06:12 this morning"* instead of *"the gate is closed"* for the ninetieth time. A window that finds everything exactly as it was is counted as a confirmation and never written to Rooms; night vision is detected from the pixels so an orange car does not become a white one after dark. Transitions accumulate into habits — *"the orange car usually leaves around 21:00"* — which is what makes an absence (*it has not left today*) sayable at all. See `GET /cameras/state` and the assistant's `camera_scene` tool.

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
TOMORROW_PLANNER_INTERVAL_SECONDS=900
MAINTENANCE_WINDOW_START_HOUR=4
MAINTENANCE_WINDOW_END_HOUR=5
AGENT_JOB_STATE_PATH=./data/agent_jobs.json
MAINTENANCE_RETRIES=3              # a failed maintenance job retries...
MAINTENANCE_RETRY_SECONDS=900      # ...instead of waiting for tomorrow
DAILY_REPORT_CATCH_UP_SECONDS=3600 # how late a *missed* run may still happen
AGENT_CHECKIN_CATCH_UP_SECONDS=21600
```

From 04:00 through 04:59 the orchestrator admits only maintenance-window jobs.
Ordinary interval jobs and agent check-ins remain due and catch up after 05:00;
Quran stays at 06:30. New automatic screen, camera, and mobile inference is also
suppressed for the hour so it cannot compete for the model; manual user requests
remain available. The reserved sequence is personal-memory deduplication at
04:00, long-term graph consolidation at 04:05, verification of personal memory
against the user's own reflection answers at 04:10, generation of an adaptive deep
reflection questions at 04:25, and clip-retention cleanup at 04:45.

#### When the PC was off at 04:00

The scheduler keeps its run history in `AGENT_JOB_STATE_PATH`, so a machine that
was asleep or shut down through the maintenance hour does not lose the night: on
the next start each job is simply overdue and runs on the first tick, then
returns to its normal slot. A multi-day gap is caught up once, not once per day
missed, and catch-up work is admitted one maintenance job per tick so it cannot
hold up deadline reminders while the user is at the machine. A job that fails —
the graph or the model not up yet at boot — retries on `MAINTENANCE_RETRY_SECONDS`
before falling back to tomorrow.

The two jobs whose output is pinned to a date bound their catch-up instead: a
23:30 report caught up the next afternoon would review the wrong day, so it is
dropped after `DAILY_REPORT_CATCH_UP_SECONDS` and a morning check-in after
`AGENT_CHECKIN_CATCH_UP_SECONDS`.

`GET /orchestrator/status` reports every job's schedule, next due time, run,
failure and deferral counts, last error, any pending retry, and the current
budget. `POST /orchestrator/jobs/{job_id}/run` triggers one immediately — the
schedule is bypassed, the delivery budget is not.

### What is running right now

One GPU, many tenants: screen and camera extraction, room chat, reflections,
the scheduled agents, speech synthesis and transcription all queue for the same
device, and a slow machine used to be explainable only from the server log.
`utils/jobs.py` is a process-wide board every one of them opens a job on —
inference is registered by wrapping the VLM client itself, so a new call site
cannot forget — and `GET /jobs` serves it: what is in flight (with the frame
count that explains a slow extraction and how long it has been running), what
finished just before, and which agents are due next. The home screen polls it
every two seconds; a streaming reply stays on the board until its last token,
not just until the request opens.

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
MEMORY_CONSOLIDATION_AT=04:05
MEMORY_REFINEMENT_ENABLED=true
MEMORY_REFINEMENT_AT=04:00
MEMORY_DUPLICATE_THRESHOLD=0.88
DAILY_REFLECTION_ENABLED=true
DAILY_REFLECTION_AT=04:25
REFLECTION_MEMORY_AUDIT_ENABLED=true
REFLECTION_MEMORY_AUDIT_AT=04:10
REFLECTION_AUDIT_BATCH=25
REFLECTION_AUDIT_CANDIDATES=14
PRODUCT_REVIEW_ENABLED=true
PRODUCT_REVIEW_AT=mon 07:00
MEMORY_QUARANTINE_DAYS=45
PROJECT_DORMANT_DAYS=21
```

`memory/refinement.py` conservatively merges near-identical personal facts while
preserving evidence and never merging same-topic facts whose values differ.
`agents/daily_reflection.py` stores the daily exercise and answers in SQLite.
Generation uses adaptive thinking and validated structured output: 8–20
questions selected from whatever is genuinely alive in the evidence, with
open-vocabulary category labels rather than a mandatory daily checklist. The
dedicated Daily Reflection room lets the user answer
throughout the day or generate manually if the overnight run was missed.

### Answers as ground truth

Almost everything this system believes about the user was inferred by watching a
screen. The reflection answers are the exception: he wrote them himself, on
purpose, having spent real time on the question. They are treated accordingly.

* **Retrieval, not a dump.** `DailyReflectionStore.prompt_context()` ranks
  answers against the question actually being asked and renders them with one
  fixed precedence rule — what he wrote outranks anything the system inferred.
  Every agent room, the Life Studio coach, the proactive narrator, the daily and
  period reports, the planner, and the main
  assistant paste the same block; ordinary topic rooms only get an answer that
  genuinely matches, rather than a loose one.
* **Repair, not only recall.** `memory/verification.py` spends each answer on
  personal memory: candidate facts go to the model with the answer as ground
  truth, and the verdicts confirm what still holds, correct what drifted,
  retract what was never true, and learn what memory did not hold at all. It
  replaced the old extraction pass, which could only ever add.
* **Guards.** A verdict may only touch a fact that was supplied as a candidate;
  a correction or retraction must quote the answer, and the quote must really
  appear in it; every applied change is logged in `personal_fact_audits` and
  individually revertible from the reflection screen or
  `POST /memory/audits/{audit_id}/revert`.
* **Weekly, the answers are read back as product feedback.** Monday 07:00,
  `agents/product_review.py` reads the week that just finished and asks what it
  implies about *this application* — where the user needed something and could
  not get it, had to re-derive what the app already knew, or was told something
  unhelpful. Every suggestion must quote the answer it came from; a life problem
  is not a missing feature; the current app surface is in the prompt so nothing
  already built gets proposed again; and a quiet week correctly produces no
  suggestions. The user marks each one planned/building/shipped/dismissed, those
  verdicts survive a regenerate, and next week's review is shown them so a
  dismissed idea does not come back. `GET /reflections/weekly-review`,
  `POST /reflections/weekly-review/generate`,
  `PUT /reflections/suggestions/{id}`; the reflection screen's lightbulb opens it.
* **The loop closes.** The audit runs at 04:10, before the 04:25 generation, so
  the new set is built from memory the answers have already corrected and is
  told which beliefs are still unconfirmed — those get at least three of the
  daily question set. Prompts mark each fact `CONFIRMED`/`UNCONFIRMED`, the screen
  extractor is shown the confirmed set so it stops re-guessing it, and
  `GET /reflections/insights` reports what the answering has bought.

Question generation, planner generation, nutrition analysis, focus
recaps, daily/long-term reports, and scheduled/manual agent check-ins use the
enforced intelligent-generation profile: Claude Agent SDK, adaptive thinking
without an explicit thinking-token ceiling, `effort=high`, and every configured
local/MCP tool. These calls fail closed when Claude is unavailable instead of
silently substituting the direct model. Finite model-turn and tool-call limits
remain as runaway-process safety guards; they do not cap thinking tokens.

The written activity report runs at `effort=xhigh`
(`REPORT_NARRATION_EFFORT`), because it is the one generation asked for a
judgement rather than a summary. It is given the whole period — every
application and project row rather than a top slice, each day's per-activity
split, the hour-of-day profile, the ranked claims with the count that was
discarded, the deterministic report itself to disagree with, his reflection
answers, and the previous fourteen days of its own reports with the scores it
gave them. What it returns is its own: it chooses the sections, merges labels
the string-matching cannot (recording each merge with its reason), scores the
period on dimensions it picks, and gives one overall 0-100 score calibrated
against that fortnight. Each report is stored on its `:Day`, so the series it
is calibrated against is the series it built. The Reports view keeps the two
apart in tabs — arithmetic that cannot be wrong about what it measured, and
judgement that is allowed to contradict it.

`GET /memory/long-term` returns the coarse tier — a fixed-size read whatever the
history length. `POST /memory/consolidate` runs the pass now, or backfills
history with `?start_date=&end_date=`. Agent rooms reach the same tier through
the `graph_long_term_memory` tool.

### Claude Code room runtime

When `AGENT_RUNTIME_ENABLED=true`, every room runs through Claude Agent SDK.
Claude Code owns the multi-step model/tool loop and talks to the configured
local Anthropic-compatible endpoint. External and in-process tools are exposed
through MCP. Disabling the runtime globally retains the direct
`FastAPI -> AsyncOpenAI -> local model` emergency fallback.

Install the optional layer with:

```powershell
pip install -r requirements-agent.txt
```

The SDK's MCP dependency tracks the newest Starlette. Starlette 1.x drops the
router arguments FastAPI 0.118 passes, so `requirements-agent.txt` pins a
compatible range.

Each room persists a toolset allowlist and an execution-depth profile: **Quick**
caps the loop at three model/tool turns, **Investigate** uses the normal room
budget, and **Act** is intended for explicitly granted writable workspace tools.

```dotenv
AGENT_RUNTIME_ENABLED=true
ANTHROPIC_BASE_URL=http://localhost:8888
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=Qwen/Qwen3.6-35B-A3B
MCP_CONFIG_PATH=./mcp.config.json
AGENT_WORKSPACE_ROOT=C:/d/agent_workspaces
RESEARCH_WORKSPACE=D:/research
RESEARCH_AGENT_REQUEST_LIMIT=32
RESEARCH_AGENT_TOOL_CALLS_LIMIT=64
```

#### Tools

Rooms see two kinds of tool:

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
  absolute path. `RESEARCH_WORKSPACE` gives the built-in Research agent a
  dedicated durable location (the recommended Windows value is `D:/research`).
  It bootstraps a numbered topic-to-submission timeline, literature folders for
  inbox/unread/reading/read/included/excluded/duplicate papers, canonical CSV
  search and screening logs, a living research plan, a proposal outline, and
  `literature_review.html`. Existing files are never replaced by bootstrap.
  The agent keeps the paper CSV, on-disk classification and HTML dashboard in
  sync. The Playwright MCP output directory is also placed under that workspace
  so downloaded papers remain available to its filesystem tools.
* **The activity graph** (`agents/graph_tools.py`) — an in-process Claude SDK
  MCP server backed directly by this application's memory. It exposes
  `mcp__graph__search_memory`, `mcp__graph__recent_activity`,
  `mcp__graph__event_detail`, `mcp__graph__day_summary`,
  `mcp__graph__long_term_memory` and `mcp__graph__room_overview`. All
  are read-only and trim storage columns and long text before the results reach
  the model. Topic, Screen, and Camera rooms are bound server-side to their own
  room; Daily and built-in advisor rooms intentionally retain shared-memory
  access. Global day and long-term summaries are exposed only to shared-memory
  rooms.

Claude Code owns each MCP session for the duration of the agent run.

#### Responses

Room responses add `execution: agent` plus Claude session/message IDs and the
tool call/output trace. Graph evidence discovered during the tool loop is merged
into normal citations and persisted with the assistant message. The built-in
Research room defaults to Act depth with graph, browser MCP, and
filesystem MCP access, and carries a reproducible SLR/PRISMA workflow prompt.
Research uses its own larger request/tool budget because a review may require
many search, download, read, and write turns. Agent chat uses the NDJSON endpoint
to show live activity: selected toolsets, tool calls with redacted argument
summaries, tool completion, drafting, and periodic still-working heartbeats.
Each room can override **Model request limit** and **Tool call limit** in
Room settings; blank values inherit the normal or Research defaults.
Streaming room chat retains NDJSON; after the tool loop completes it emits the
final text, citations, and tool trace. `GET /agent-runtime/status` reports limits,
native toolset count, configured MCP server names, and the selectable room tool
catalog without connecting to any server. It also exposes the enforced
`intelligent_generation` profile so the runtime settings are inspectable.

#### Structured outputs

Calls that need data rather than prose pass a Pydantic model as `output_type`
(see `agents/schemas.py`). Both paths honour it: Claude structured output
validates on the agent path, and the direct path validates the same model after
extracting JSON from the reply. The next-day planner uses `PlanProposal` this way instead of
scraping dictionaries out of free text. There is no evaluation schema: no model
is allowed to decide that a task was completed.

Tomorrow tasks can also be created manually before a generated plan exists or
after tracking starts. An unfinished task is never dropped: each generated plan
first inherits every still-open task from earlier days, with no lookback window,
and only the user can complete or delete one. How long each task has been open
(`days_open`, `carried_count`, `first_planned_on`) is the accountability signal
every personal agent room is given. Each task may carry a local deadline, a configurable
number of reminders, and a configurable delay between them. Overdue reminders
are durable notifications and are spoken on Android through system TTS; active
desktop clients use Kokoro. The user is asked to record the blocker or choose a
new deadline. Reminder history and those responses are persisted and supplied
to Tomorrow, Roaster, and Life Studio for later accountability analysis.

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
  observations. Every five minutes the Claude agent combines live context,
  graph-linked project/entity memory, personal preferences, active focus goals,
  and past reactions. It privately explores several imaginative angles and
  self-scores the strongest one for relevance, novelty, usefulness, and
  insightfulness. Code enforces the score threshold and silently discards weak,
  repetitive, or operationally noisy candidates (logs, errors, installers,
  running processes, telemetry, and graphs).

* **Evidence you can watch and question**
  Anything the assistant raises unprompted — a proactive nudge, an important or
  critical alert — carries the footage it was made from: a low-resolution,
  sped-up clip of the exact capture window (`GET /clips/<id>`, playable inline
  in the app). Those clips also answer follow-up questions from the video
  itself (`POST /clips/<id>/ask` → "was he carrying anything?"), so a claim
  about something you did not see can be checked instead of just believed.
  Clips are retained for 30 days and expired once daily at 04:45; referenced
  clips are protected ahead of ordinary clips if the storage cap is reached
  (`sources/clips.py`).

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
| **VLM Engine** | Qwen3.6 35B-A3B (llama.cpp) | 2 min temporal image sequence per inference |
| **ASR** | Parakeet / Whisper | Ultra-low latency / Multilingual |
| **Memory** | Vector Store + Reranker | Full-day context retention |

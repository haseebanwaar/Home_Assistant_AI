# agents/

## Active

- **`orchestrator.py`** — the layer that connects everything below. One
  `Orchestrator` holds every scheduled agent as an `AgentJob` and drives them
  from a single tick, replacing the per-agent `while True: sleep` loops that
  used to live in `app.py`. It owns three things worth knowing:
  - **Schedules** — `Interval`, `DailyAt` (which catches up a slot the machine
    slept through, but never fires on the spot at boot), and `Manual`.
  - **`DeliveryBudget`** — the arbitration. Everything that addresses the user
    unprompted claims from one shared throttle, including `ProactiveNarrator`
    running off a capture thread. A job denied a slot is *deferred*, not
    dropped: it stays due and delivers once the user is not being interrupted.
    A job that claims a slot and then stays silent gives it back.
  - **`OrchestratorContext`** — the day's metrics, claims and focus state read
    once per tick and shared by every job, rather than eight agents issuing the
    same query.

  Jobs are registered in `app.py:register_agent_jobs()`: clip retention, the
  nightly report, the planner lifecycle, memory consolidation, and one check-in
  per personal agent. `GET /orchestrator/status` reports each job's schedule,
  next due time, run/failure/deferral counts and last error;
  `POST /orchestrator/jobs/{job_id}/run` triggers one by hand.

- **`personal_agents.py`** — the built-in agent rooms. Each carries a `check_in`
  prompt and now a `check_in_at` time; the orchestrator runs the prompt at that
  hour, so the agents accompany the day instead of waiting to be pressed.
  Times are spread out (`06:30` Quran, `08:00` Wisdom, `09:15` Motivational,
  `18:00` PhD, `21:30` Roaster) and overridable through
  `AGENT_CHECKIN_SCHEDULE`; blank means manual only, which is what Research
  (expensive, tool-driven) and Tomorrow (the planner already runs it) use.

- **`proactive.py`** — `ProactiveNarrator`. The initiative layer receives
  desktop, mobile, and camera observations, combines them with retrieved memory,
  focus context, and reaction history, then lets the VLM decide whether and how
  to speak unprompted. It is enabled by default through `PROACTIVE_ENABLED`;
  cooldown and duplicate suppression pace delivery across capture threads, and
  the shared `DeliveryBudget` paces it against the scheduled agents. The budget
  is checked before the VLM call (a nudge that could not be delivered costs
  nothing) and claimed again at delivery, so a check-in that landed while the
  model was drafting wins.

## `_parked/` (not wired, kept for reference)

The original event-bus + autogen multi-agent layer. It is **not** imported or
run anywhere and was largely non-functional (inconsistent `AsyncAgent.__init__`
signatures that raised `TypeError`, calls to a non-existent `vlm.ainvoke`, an
`EventBus.run_forever` loop fed by a queue nobody wrote to, an empty
`agent_group_spawner`). Parked rather than deleted so the ideas aren't lost.

If you revive any of it: fix the `AsyncAgent` base signature, replace
`vlm.ainvoke` with the OpenAI-style `client.chat.completions.create`, and either
feed `EventBus.queue` or drop the queue and keep the direct `publish`. For this
single-user POC, prefer extending `proactive.py` and `tools/registry.py` over
reintroducing the framework.

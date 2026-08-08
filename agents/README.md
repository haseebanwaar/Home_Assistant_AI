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
  - **`JobStateStore`** — the run history in `data/agent_jobs.json`. Without it
    every `last_run` died with the process, so a PC that was off through the
    04:00 maintenance hour came back with a clean slate and waited for the
    *next* 04:00 — the night was silently skipped. With it the jobs are simply
    overdue and run on the first tick after startup. Two guards keep that from
    misfiring: `catch_up_seconds` drops a missed run whose output is pinned to
    the wall clock (a 23:30 report caught up at noon would review the wrong
    day), and outside the maintenance hour only one maintenance job is admitted
    per tick, so a whole night of catch-up cannot hold up deadline reminders.
    A failed run also retries on `retry_delay_seconds` up to `max_retries`
    rather than costing the whole day.

  Jobs are registered in `app.py:register_agent_jobs()`: clip retention, the
  nightly report, the planner lifecycle, memory consolidation, and one check-in
  per personal agent. `GET /orchestrator/status` reports each job's schedule,
  next due time, run/failure/deferral counts, pending retry and last error;
  `POST /orchestrator/jobs/{job_id}/run` triggers one by hand.

- **`personal_agents.py`** — the built-in agent rooms. Each carries a `check_in`
  prompt and now a `check_in_at` time; the orchestrator runs the prompt at that
  hour, so the agents accompany the day instead of waiting to be pressed.
  Times are spread out (`06:30` Quran, `08:00` Wisdom, `09:15` Motivational,
  `18:00` PhD, `21:30` Roaster) and overridable through
  `AGENT_CHECKIN_SCHEDULE`; blank means manual only, which is what Research
  (expensive, tool-driven) and Tomorrow (the planner already runs it) use.

- **`horizons.py`** — the only module that reasons above the altitude of a day.
  Every other agent here answers about today, this week at the most, which means
  a year of captured life ends up used a day at a time forever. This one holds
  six windows — week, month, quarter, half-year, year, lifelong — and for each
  writes a reflection plus a forecast for the next window of the same size.
  Three mechanisms make it compound rather than plateau:
  - **Hierarchy.** A year review reads its four quarter reviews, which read
    their months, which read their weeks (`CHILD_HORIZON`). Context stays
    bounded however long the history gets, so a lifelong review costs about what
    a weekly one costs, and each tier stands on distilled material.
  - **Graded forecasts.** Every prediction carries a due date. Once it passes,
    `grade_due_horizon_predictions` judges it against evidence as hit / miss /
    partial / unclear, and `HorizonStore.calibration()` turns that into a hit
    rate per confidence band which is fed back into the next forecast. Unclear
    leaves the denominator — it is a drafting fault, not a wrong answer.
  - **Threads.** A pattern is named once and then updated with a dated note each
    period, keyed by the slug of its name, so the same thread in March and in
    November is one trajectory. A thread the user closes stays closed.

  `app.py` runs one daily job (`horizon-reviews`, 07:30) that grades what came
  due and then writes whichever windows have actually closed, shortest first and
  a couple per run — six can close on the same morning. Nothing is written about
  a window that has not finished. The user's verdict on any forecast or thread
  overrides the model's through `PUT /horizons/predictions/{id}` and
  `PUT /horizons/threads/{id}`, and survives a regenerate.

- **`quran_study.py`** — the Quran room's store. The room used to ask the chat
  endpoint for a JSON study guide and keep the result in the client's
  `SharedPreferences`, so a reply truncated by the 700-token chat budget parsed
  into an empty guide and the vocabulary, classical-commentary and
  modern-reflection cards rendered blank with no error anywhere. The report is
  now `agents.schemas.QuranStudyGuide`, whose minimum lengths make an empty
  section a validation failure that `generate_quran_study_guide` retries once
  rather than stores. Three things follow from having a real store:
  - **Reports are saved when written**, not when the user remembers to press
    "complete reading", so a guide he read and closed no longer takes its words
    with it. Regenerating a passage on the same day replaces the report and
    keeps the reflection he wrote against it.
  - **One deck, not one list per session.** Words are lifted out of every guide
    and deduplicated on normalized Arabic (`word_key` drops harakat and folds
    hamza carriers), so the same word met in three passages is one entry that
    gained three senses.
  - **The recall mark is his.** Each word is `learning` or `known`, set only
    through `PUT /quran/vocabulary/{word_id}`; no regeneration touches it, and
    `study_context` feeds the deck back to the model so it stops re-teaching
    what he already remembers. `QuranStudyStore.canvas()` publishes the journey
    to the room canvas, which is where Daily Reflection, Horizons and the
    Roaster's accountability metrics read it.

- **`calendar.py`** — `CalendarStore`, the only record of what the user *meant*
  to be doing. Every other store holds evidence of what happened, which left
  each accountability room one story to tell about a quiet week — avoidance —
  and it told that story about weeks spent ill or travelling. Two things live
  here:
  - **A routine** is the shape of an ordinary week: blocks tied to weekdays, not
    dates, written once instead of re-entered every Monday. It is the baseline a
    gap is measured against, and the prompt block labels it as intention, never
    as evidence that the hour was actually spent.
  - **An entry** is a dated fact — a holiday, a trip, a deadline, an illness —
    and it can change what the day *is*. `repeat` (`weekly` / `monthly` /
    `yearly`) writes a standing commitment once; recurrence is never
    retroactive, and a monthly entry anchored to the 31st skips the months that
    have no 31st rather than sliding onto a date he never wrote.

  Both carry a **label**: his own word for the thing, free text, with no fixed
  vocabulary anywhere in the file and no meaning derived from it. Every reader
  is a Claude agent with the graph in reach, so "aqiqah" or "hospital with dad"
  is carried through to the prompt verbatim for the room to understand — a
  closed list could only refuse the word or file it under "other". The editor
  offers the labels he has already used as chips (`GET /calendar` → `labels`)
  and accepts anything else he types.

  The one field the app acts on mechanically is `expectation` (`normal` /
  `reduced` / `none`), and **only when he sets it himself** — nothing derives it
  from a label. When several entries on a date declare one, **the most forgiving
  one wins**, because a meeting accepted last week does not make him less ill
  today. A blank one is not a claim that the day was ordinary: `day()` returns
  `declared: false`, and the prompt blocks list those days separately with his
  words attached and hand the reading to the room.

  Three rules hold it honest: an expectation explains a gap but never invents an
  achievement; nothing infers an entry, because a wrong guess would silently
  forgive real avoidance; and a suspended day changes the stance for *that* day
  only, leaving earlier ordinary days as accountable as they were. A label alone
  never silences a check-in either — reading his words is the room's job, and it
  has to say out loud how it read them.

  It reaches the model in three places — `prompt_context()` in every personal
  agent's room grounding, `planning_context()` in tomorrow's plan generation
  (which also refuses to propose a normal list for a day marked sick or away),
  and `check_in_directive()`, which bends the unprompted scheduled check-in,
  the one turn where nobody is present to correct a 06:00 roast about a routine
  he had already written off. REST under `/calendar`; the editor is
  `front_end/lib/calendar/calendar_screen.dart`.

- **`proactive.py`** — `ProactiveNarrator`. The initiative layer receives
  desktop, mobile, and camera observations, combines them with retrieved memory,
  focus context, and reaction history, then gives the Claude Agent SDK graph
  tools to search memory further. Claude generates and self-scores a candidate;
  deterministic quality and operational-noise gates publish only strong
  insights. `PROACTIVE_INTERVAL_SECONDS` defaults to 300 seconds. Duplicate
  suppression and the shared `DeliveryBudget` still pace delivery against the
  scheduled agents. The budget is checked before the Claude run and claimed
  again at delivery, so a check-in that landed while it was drafting wins.

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

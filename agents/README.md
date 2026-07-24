# agents/

## Active

- **`proactive.py`** — `ProactiveNarrator`. The one working agent: given a
  minute's screen description it asks the VLM whether to speak an unprompted
  insight, with a cooldown. Wired in `app.py` (gated by `PROACTIVE_ENABLED`)
  and driven by `sources/screen.py`'s `insight_callback`.

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

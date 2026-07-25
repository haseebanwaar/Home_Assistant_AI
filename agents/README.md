# agents/

## Active

- **`proactive.py`** — `ProactiveNarrator`. The initiative layer receives
  desktop, mobile, and camera observations, combines them with retrieved memory,
  focus context, and reaction history, then lets the VLM decide whether and how
  to speak unprompted. It is enabled by default through `PROACTIVE_ENABLED`;
  cooldown and duplicate suppression pace delivery across capture threads.

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

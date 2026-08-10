"""Tests for focus recaps, room hygiene, nudge suppression and prompt learning.

All pure logic — no Neo4j, Qdrant or VLM required.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

from agents.proactive import (
    INSIGHT_REQUEST_LIMIT, INSIGHT_TOOL_CALLS_LIMIT, ProactiveNarrator,
    _SYSTEM_PROMPT, _repeats_opening, _similar,
)
from agents.schemas import ProactiveInsightDecision
from memory.extraction.prompts import _naming_block, build_system_prompt
from memory.rooms.hygiene import merge_suggestions, stale_rooms
from memory.summary import focus_recap


# -- focus recap -----------------------------------------------------------
def _event(summary, start, end, application="Code"):
    return {"summary": summary, "span_start": start, "span_end": end,
            "application": application}


def test_labels_are_parsed_from_a_json_array():
    raw = 'Sure!\n[{"n": 1, "label": "on"}, {"n": 2, "label": "off"}]'
    assert focus_recap.parse_labels(raw, 2) == {1: "on", 2: "off"}


@pytest.mark.parametrize("raw", ["not json at all", "", '[{"n": 99, "label": "on"}]',
                                 '[{"n": 1, "label": "maybe"}]'])
def test_unusable_labels_are_dropped(raw):
    assert focus_recap.parse_labels(raw, 2) == {}


def test_minutes_follow_real_spans_not_event_counts():
    events = [_event("wrote the retry loop", 0, 600),      # 10 min on
              _event("watched a video", 600, 900, "Chrome")]  # 5 min off
    breakdown = focus_recap.apply_classification(
        "fix the RTSP retry loop", events, {1: "on", 2: "off"}, 0, 900)

    assert breakdown["on_task_minutes"] == 10.0
    assert breakdown["off_task_minutes"] == 5.0
    assert breakdown["on_task_pct"] == 67
    assert breakdown["distractions"] == [{"app": "Chrome", "minutes": 5.0}]


def test_event_time_is_clipped_to_the_session_window():
    """An event straddling the boundary must not inflate the session."""
    events = [_event("long running", -600, 1200)]
    breakdown = focus_recap.apply_classification("goal", events, {1: "on"}, 0, 600)
    assert breakdown["on_task_minutes"] == 10.0


def test_unlabelled_events_are_reported_as_unclear_not_as_failure():
    """A failed classification must not read as 'you were off task'."""
    events = [_event("something", 0, 600)]
    breakdown = focus_recap.apply_classification("goal", events, {}, 0, 600)

    assert breakdown["unknown_minutes"] == 10.0
    assert breakdown["off_task_minutes"] == 0.0
    assert breakdown["on_task_pct"] is None
    assert "Not enough judged activity" in focus_recap.format_recap(breakdown)


def test_recap_renders_goal_score_and_distractions():
    events = [_event("fixed the loop", 0, 600),
              _event("youtube", 600, 900, "Chrome")]
    breakdown = focus_recap.apply_classification(
        "fix the loop", events, {1: "on", 2: "off"}, 0, 900)
    text = focus_recap.format_recap(breakdown, planned_minutes=25, feedback="Nice.")

    assert "fix the loop" in text
    assert "67% on task" in text
    assert "of 25 planned" in text
    assert "Chrome" in text
    assert "Nice." in text


def test_empty_session_still_produces_a_recap():
    breakdown = focus_recap.apply_classification("goal", [], {}, 0, 600)
    assert breakdown["events"] == 0
    assert focus_recap.format_recap(breakdown)


# -- room hygiene ----------------------------------------------------------
NOW = 1_000_000.0
DAY = 86400


def _room(room_id, **kw):
    base = {"room_id": room_id, "name": room_id, "kind": "project", "auto": True,
            "pinned": False, "events": 5, "notes": 0, "messages": 0,
            "last_event_at": NOW - 40 * DAY, "active_minutes": 5}
    base.update(kw)
    return base


def test_cold_and_thin_auto_rooms_are_archive_candidates():
    stale = stale_rooms([_room("project:one-off")], now=NOW)
    assert [r["room_id"] for r in stale] == ["project:one-off"]
    assert stale[0]["idle_days"] == 40


def test_a_long_running_project_that_went_quiet_is_left_alone():
    """Cold but substantial — the user clearly invested in it."""
    assert stale_rooms([_room("project:big", active_minutes=600)], now=NOW) == []


def test_recently_active_rooms_are_left_alone():
    assert stale_rooms([_room("project:live", last_event_at=NOW - DAY)], now=NOW) == []


@pytest.mark.parametrize("override", [
    {"pinned": True}, {"auto": False}, {"kind": "daily"},
    {"notes": 1}, {"messages": 1},
])
def test_rooms_the_user_invested_in_are_never_touched(override):
    assert stale_rooms([_room("project:x", **override)], now=NOW) == []


def test_a_room_with_no_events_at_all_is_a_candidate():
    stale = stale_rooms([_room("project:empty", last_event_at=None, events=0)], now=NOW)
    assert stale[0]["reason"] == "no activity recorded"


def test_merge_suggestion_folds_the_smaller_room_into_the_larger():
    overlaps = [{"room_a": "project:a", "name_a": "A", "room_b": "project:b",
                 "name_b": "B", "shared": 7, "overlap_pct": 80}]
    stats = [_room("project:a", active_minutes=10),
             _room("project:b", active_minutes=200)]

    [suggestion] = merge_suggestions(overlaps, stats)

    assert suggestion["source_room_id"] == "project:a"   # smaller
    assert suggestion["target_room_id"] == "project:b"   # keeps the bigger feed
    assert "7 shared entities" in suggestion["reason"]


# -- nudge suppression -----------------------------------------------------
def test_rephrased_nudges_are_treated_as_duplicates():
    assert _similar("You have been in YouTube for six minutes now.",
                    "You've now spent six minutes in YouTube.")


def test_distinct_nudges_are_not_suppressed():
    assert not _similar("You have been in YouTube for six minutes.",
                        "Your Neo4j password is missing from the env file.")


def test_short_insight_is_not_suppressed_by_partial_containment():
    assert not _similar(
        "A smaller experiment could reveal the real constraint.",
        "The smaller experiment from Tuesday revealed a budget constraint, so "
        "reuse its measurements before redesigning the whole pipeline.")


def test_reused_opening_is_detected_independently_of_the_rest_of_the_message():
    assert _repeats_opening(
        "It looks like the build passed.",
        ["It looks as though that retry recovered."])
    assert not _repeats_opening(
        "That retry recovered.",
        ["It looks as though that retry recovered."])
    assert not _repeats_opening(
        "The smaller trial would answer this quickly.",
        ["The launch schedule leaves no margin."])


def test_proactive_prompt_uses_open_ended_model_judgment():
    assert "full judgment, imagination, and creativity" in _SYSTEM_PROMPT
    assert "Privately brainstorm several possible" in _SYSTEM_PROMPT
    assert "linked memory" in _SYSTEM_PROMPT
    assert "Adapt your tone to the moment" in _SYSTEM_PROMPT
    assert "Vary the opening words" in _SYSTEM_PROMPT
    assert "Do not proactively discuss logs" in _SYSTEM_PROMPT
    assert "critically score" in _SYSTEM_PROMPT


def test_proactive_context_includes_source_without_prescribing_a_verdict():
    narrator = ProactiveNarrator("model", client=None)
    prompt = narrator._build_prompt(
        "A delivery van stopped at the gate.", focus=None, evidence=[],
        source="camera:Driveway", context={"camera_id": "camera:driveway"})

    assert "Live observation source: camera:Driveway" in prompt
    assert "camera_id: camera:driveway" in prompt
    assert "warranted" not in prompt


def test_proactive_prompt_includes_personal_context_and_openings_to_avoid():
    narrator = ProactiveNarrator("model", client=None)
    prompt = narrator._build_prompt(
        "The user is debugging a camera stream.", focus=None, evidence=[],
        personal_context="The user prefers direct technical suggestions.",
        recent_texts=["It looks like the stream stalled again."])

    assert "prefers direct technical suggestions" in prompt
    assert "it looks like the" in prompt
    assert "do not repeat or closely imitate" in prompt
    assert "tone that fits the live moment" in prompt
    assert "graph-memory tools" in prompt
    assert "installers/MSI activity" in prompt


def test_quality_gate_accepts_strong_useful_candidate_without_near_ceiling_scores():
    weak = ProactiveInsightDecision(
        publish=True, insight="Maybe consider a different approach.",
        relevance=4, novelty=2, usefulness=3, insightfulness=3)
    noisy = ProactiveInsightDecision(
        publish=True, insight="The MSI installer log shows another runtime error.",
        relevance=5, novelty=5, usefulness=5, insightfulness=5,
        operational_noise=True)
    useful = ProactiveInsightDecision(
        publish=True,
        insight="Those errors may expose an assumption worth testing separately.",
        relevance=4, novelty=3, usefulness=4, insightfulness=3)

    assert not ProactiveNarrator._passes_quality(weak)
    assert not ProactiveNarrator._passes_quality(noisy)
    assert ProactiveNarrator._passes_quality(useful)


def test_claude_agent_uses_graph_memory_and_only_passes_high_quality():
    class Runtime:
        def __init__(self):
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(output=ProactiveInsightDecision(
                publish=True,
                insight=("The recurring wish to simplify this project could become a "
                         "design rule: make the next feature remove one decision."),
                relevance=5, novelty=4, usefulness=5, insightfulness=5,
                memory_used=True,
                rationale="Connects the current project to a durable preference."))

    runtime = Runtime()
    narrator = ProactiveNarrator(
        "claude-model", client=None, agent_runtime=runtime,
        cooldown_seconds=0, evaluation_interval_seconds=300)

    result = asyncio.run(narrator.consider("Sketching the next project feature."))

    assert result["text"].startswith("The recurring wish")
    assert runtime.calls[0]["selected_tools"] is None
    assert runtime.calls[0]["room_id"] == "agent:proactive-insight"
    assert runtime.calls[0]["output_type"] is ProactiveInsightDecision
    assert runtime.calls[0]["thinking"] is True
    assert runtime.calls[0]["thinking_budget"] is None
    assert runtime.calls[0]["effort"] == "high"
    # The narrator is told to go and check memory before it speaks, so its
    # budget has to cover the tool loop it was asked to run. Too small and the
    # run ends on the budget instead of on an answer, which is silence.
    assert runtime.calls[0]["configured_request_limit"] == INSIGHT_REQUEST_LIMIT
    assert (runtime.calls[0]["configured_tool_calls_limit"]
            == INSIGHT_TOOL_CALLS_LIMIT)
    assert INSIGHT_REQUEST_LIMIT >= 8 and INSIGHT_TOOL_CALLS_LIMIT >= 12


def test_discarded_candidate_waits_five_minutes_before_another_evaluation(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("agents.proactive.time.time", lambda: now[0])

    class Runtime:
        def __init__(self):
            self.calls = 0

        async def run(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(output=ProactiveInsightDecision(
                publish=False, insight="", relevance=2, novelty=2,
                usefulness=2, insightfulness=2,
                rationale="No worthwhile insight."))

    runtime = Runtime()
    narrator = ProactiveNarrator(
        "claude-model", client=None, agent_runtime=runtime,
        cooldown_seconds=0, evaluation_interval_seconds=300)

    assert asyncio.run(narrator.consider("Routine work.")) is None
    now[0] += 299
    assert asyncio.run(narrator.consider("More routine work.")) is None
    assert runtime.calls == 1
    now[0] += 1
    assert asyncio.run(narrator.consider("A new five-minute window.")) is None
    assert runtime.calls == 2


def test_proactive_decision_preserves_observation_source():
    class Completions:
        async def create(self, **_kwargs):
            message = SimpleNamespace(content="The van may be your delivery.")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()))
    narrator = ProactiveNarrator("model", client, cooldown_seconds=0)

    result = asyncio.run(narrator.consider(
        "A van stopped outside.", source="camera:Driveway"))

    assert result["source"] == "camera:Driveway"
    # `kind` is the sort of nudge, `source` is where it came from. Echoing the
    # source into kind gave every camera its own kind ('camera:Driveway') and
    # logged nudges under a device name instead of a category.
    assert result["kind"] == "insight"


def test_proactive_rewrites_a_repeated_opening_instead_of_silently_dropping_it():
    class Completions:
        def __init__(self):
            self.responses = iter([
                "It looks like the same decoder fault.",
                "That decoder fault matches Tuesday's failed buffer handoff.",
            ])
            self.calls = 0

        async def create(self, **_kwargs):
            self.calls += 1
            message = SimpleNamespace(content=next(self.responses))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class Store:
        def active_focus_session(self):
            return None

        def list_nudges(self, limit=10):
            return [{"text": "It looks like the stream recovered."}]

        def recent_nudge_feedback(self, limit=8):
            return []

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    narrator = ProactiveNarrator(
        "model", client, cooldown_seconds=0, store_getter=Store)

    result = asyncio.run(narrator.consider("The decoder fault appeared again."))

    assert result["text"].startswith("That decoder fault")
    assert completions.calls == 2


def test_production_runtime_revises_repetition_instead_of_dropping_it():
    class Runtime:
        def __init__(self):
            self.calls = []

        async def run(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(output=ProactiveInsightDecision(
                    publish=True,
                    insight="It looks like the launch plan needs a smaller trial.",
                    relevance=4, novelty=3, usefulness=4, insightfulness=3))
            return SimpleNamespace(
                output="A one-day trial would test the launch assumption cheaply.")

    class Store:
        def active_focus_session(self):
            return None

        def list_nudges(self, limit=10):
            return [{"text": "It looks like the launch is ready."}]

        def recent_nudge_feedback(self, limit=8):
            return []

    runtime = Runtime()
    narrator = ProactiveNarrator(
        "claude-model", client=None, agent_runtime=runtime,
        cooldown_seconds=0, store_getter=Store)

    result = asyncio.run(narrator.consider("Reviewing the launch plan."))

    assert result["text"].startswith("A one-day trial")
    assert len(runtime.calls) == 2
    assert runtime.calls[1]["output_type"] is str


def test_recall_follows_event_entity_links_and_excludes_the_recent_window():
    class Retriever:
        def retrieve(self, *_args, **kwargs):
            assert kwargs["end"] == 100
            assert "entity" in kwargs["kinds"]
            return [{
                "kind": "event", "id": "seed", "title": "Code",
                "text": "Worked on the Atlas decoder.", "ts": 50,
            }]

    class Store:
        def entities_for_events(self, event_ids):
            assert event_ids == ["seed"]
            return {"seed": [{"name": "Atlas", "type": "project"}]}

        def entity_detail(self, entity_id):
            assert entity_id == "Atlas"
            return {
                "claims": [{
                    "claim_id": "claim-old",
                    "text": "The ring buffer owns retries.",
                    "last_seen": 40,
                }],
                "events": [{
                    "event_id": "recent",
                    "summary": "Changed the decoder today.",
                    "application": "Code",
                    "span_start": 200,
                }],
            }

    narrator = ProactiveNarrator(
        "model", client=None, retriever=Retriever(), store_getter=Store)
    recalled = narrator._recall("Debugging the Atlas decoder", now=1000)

    assert recalled[0]["entities"][0]["name"] == "Atlas"
    assert any(
        item.get("relationship") == "linked through Atlas"
        and item.get("text") == "The ring buffer owns retries."
        for item in recalled)
    assert not any(item.get("id") == "recent" for item in recalled)


# -- corrections teaching the extractor ------------------------------------
def test_naming_corrections_appear_in_the_extraction_prompt():
    prompt = build_system_prompt(None, naming_hints=[("Qwen3 VL", "qwen3-vl")])
    assert 'write "qwen3-vl" (not "Qwen3 VL")' in prompt


def test_no_hints_leaves_the_prompt_unchanged():
    assert build_system_prompt(None) == build_system_prompt(None, naming_hints=[])


def test_noop_and_blank_hints_are_ignored():
    assert _naming_block([("same", "Same"), ("", "x"), ("y", "")]) == ""

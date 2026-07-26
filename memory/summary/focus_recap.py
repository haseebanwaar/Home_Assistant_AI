"""The Coach at focus-session granularity.

`format_report` in coach.py answers "how was my day"; this answers "did I do the
thing I said I'd do", which is far more actionable because the user declared a
goal up front. That declaration is the whole point: it is the only signal that
reliably says WHEN an interruption is welcome and WHAT counts as off-task.

Split into a deterministic half and an LLM half, like coach.py:
- `classify_prompt` asks the model to label each event on/off-task vs the goal;
- `apply_classification` turns those labels into minutes using real event spans;
- `format_recap` renders the result as markdown.

Events are attributed to a session by time overlap rather than by tagging them
at capture time, so a recap can be produced for sessions that ran before this
existed, and re-running it picks up late-arriving events.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger("home_assistant")

ON, OFF, UNKNOWN = "on", "off", "unknown"


def classify_prompt(goal, events):
    """Ask for one on/off-task label per event, as JSON."""
    listing = "\n".join(
        f"{index}. {(event.get('summary') or '').strip()[:300]}"
        for index, event in enumerate(events, start=1))
    return (
        "The user started a focus session with this stated goal:\n"
        f"  {goal}\n\n"
        "Below are the activities captured from their screen during that session. "
        "For each one, decide whether it served the stated goal.\n\n"
        f"{listing}\n\n"
        'Return ONLY a JSON array like [{"n": 1, "label": "on"}, {"n": 2, "label": "off"}]. '
        f'"{ON}" = served the goal (including necessary support work such as reading '
        f'docs, debugging or running tests for it). "{OFF}" = unrelated. '
        f'"{UNKNOWN}" only when the activity is genuinely too vague to judge. '
        "No prose, no code fences."
    )


def parse_labels(raw_text, count):
    """Parse the model's label array into {index: label}; tolerant of junk."""
    text = (raw_text or "").strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return {}
    try:
        rows = json.loads(match.group(0))
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("focus recap: unparseable labels (%s)", exc)
        return {}
    labels = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("n"))
        except (TypeError, ValueError):
            continue
        label = str(row.get("label", "")).strip().lower()
        if 1 <= index <= count and label in (ON, OFF, UNKNOWN):
            labels[index] = label
    return labels


def _seconds(event, start, end):
    """Event duration clipped to the session window."""
    span_start = max(float(event.get("span_start") or start), start)
    span_end = min(float(event.get("span_end") or span_start), end)
    return max(0.0, span_end - span_start)


def apply_classification(goal, events, labels, start, end):
    """Combine labels with real spans into an on/off-task breakdown."""
    buckets = {ON: [], OFF: [], UNKNOWN: []}
    seconds = {ON: 0.0, OFF: 0.0, UNKNOWN: 0.0}
    for index, event in enumerate(events, start=1):
        label = labels.get(index, UNKNOWN)
        buckets[label].append(event)
        seconds[label] += _seconds(event, start, end)

    tracked = seconds[ON] + seconds[OFF]
    return {
        "goal": goal,
        "on_task_minutes": round(seconds[ON] / 60, 1),
        "off_task_minutes": round(seconds[OFF] / 60, 1),
        "unknown_minutes": round(seconds[UNKNOWN] / 60, 1),
        # Share of JUDGED time only — unknowns must not silently read as failure.
        "on_task_pct": round(100 * seconds[ON] / tracked) if tracked else None,
        "on_task": [e.get("summary") for e in buckets[ON]],
        "off_task": [e.get("summary") for e in buckets[OFF]],
        "distractions": _distractions(buckets[OFF]),
        "events": len(events),
        "elapsed_minutes": round((end - start) / 60, 1),
    }


def _distractions(off_events):
    """Apps that pulled the user away, by time, biggest first."""
    by_app = {}
    for event in off_events:
        app = (event.get("application") or "").strip()
        if not app:
            continue
        span = float(event.get("span_end") or 0) - float(event.get("span_start") or 0)
        by_app[app] = by_app.get(app, 0.0) + max(0.0, span)
    ranked = sorted(by_app.items(), key=lambda kv: kv[1], reverse=True)
    return [{"app": app, "minutes": round(seconds / 60, 1)} for app, seconds in ranked[:5]]


def format_recap(breakdown, planned_minutes=None, feedback=None):
    """Render the breakdown as markdown for posting into a room."""
    lines = [f"# Focus recap — {breakdown['goal']}", ""]
    elapsed = breakdown["elapsed_minutes"]
    planned = f" of {planned_minutes} planned" if planned_minutes else ""
    lines.append(f"**{elapsed:.0f} min elapsed**{planned} · {breakdown['events']} activities")

    pct = breakdown["on_task_pct"]
    if pct is None:
        lines.append("_Not enough judged activity to score this session._")
    else:
        lines.append(
            f"**{pct}% on task** — {breakdown['on_task_minutes']:.0f} min on, "
            f"{breakdown['off_task_minutes']:.0f} min off"
            + (f", {breakdown['unknown_minutes']:.0f} min unclear"
               if breakdown["unknown_minutes"] else ""))

    if breakdown["on_task"]:
        lines.append("\n## Toward the goal")
        for summary in breakdown["on_task"][:8]:
            lines.append(f"- {summary}")

    if breakdown["distractions"]:
        lines.append("\n## Pulled away by")
        for item in breakdown["distractions"]:
            lines.append(f"- {item['app']} — {item['minutes']:.0f} min")

    if feedback:
        lines.append(f"\n## Coach\n{feedback}")
    return "\n".join(lines)


def feedback_prompt(breakdown):
    """Short, specific closing note on the session."""
    return (
        "You are a supportive but honest focus coach. The user just finished a "
        "focus session. In 2-3 sentences, say whether they achieved the stated "
        "goal, name the single biggest thing that pulled them away (if any), and "
        "give one concrete suggestion for the next session. Be specific to this "
        "data and do not invent activities.\n\n"
        f"Goal: {breakdown['goal']}\n"
        f"Elapsed: {breakdown['elapsed_minutes']:.0f} min\n"
        f"On task: {breakdown['on_task_minutes']:.0f} min "
        f"({breakdown['on_task_pct']}%)\n"
        f"Off task: {breakdown['off_task_minutes']:.0f} min\n"
        + ("Distractions: " + ", ".join(
            f"{d['app']} {d['minutes']:.0f}m" for d in breakdown["distractions"])
           if breakdown["distractions"] else "No notable distractions.")
        + ("\nWork done: " + "; ".join(
            str(s) for s in breakdown["on_task"][:6]) if breakdown["on_task"] else "")
    )

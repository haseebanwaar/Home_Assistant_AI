"""Phase 3 — the Coach: a daily productivity report + feedback.

`format_report` renders deterministic metrics (time per activity/app/project,
focus score, switches) into markdown. `coach_prompt` turns those metrics plus the
day's accomplishments (claims) into a prompt for the LLM to write a short,
specific feedback note. The endpoint combines both and posts the result into the
Daily room as a 'coach' message.
"""
from __future__ import annotations


def _bar(minutes, total):
    if not total:
        return ""
    n = int(round(10 * minutes / total))
    return "#" * n + "-" * (10 - n)


def format_report(metrics, claims=None, entities=None):
    date = metrics.get("date")
    active = metrics.get("active_minutes", 0)
    lines = [f"# Daily report — {date}", ""]
    lines.append(
        f"**{active:.0f} min active** · {metrics.get('sessions', 0)} sessions · "
        f"{metrics.get('events', 0)} events · focus score **{metrics.get('focus_score', 0)}/100**")
    lines.append(
        f"_avg block {metrics.get('avg_event_seconds', 0) / 60:.1f} min · "
        f"longest {metrics.get('longest_block_seconds', 0) / 60:.1f} min · "
        f"{metrics.get('switches', 0)} context switches "
        f"({metrics.get('switches_per_hour', 0)}/hr)_")
    lines.append("")

    total = sum(a["minutes"] for a in metrics.get("by_activity", [])) or 1
    lines.append("## Time by activity")
    for a in metrics.get("by_activity", []):
        lines.append(f"- {a['activity']:<14} {a['minutes']:>5.0f} min  {_bar(a['minutes'], total)}")

    if metrics.get("by_project"):
        lines.append("\n## Projects")
        for p in metrics["by_project"]:
            lines.append(f"- {p['project']} — {p['minutes']:.0f} min")

    if metrics.get("by_app"):
        lines.append("\n## Apps")
        for a in metrics["by_app"][:6]:
            lines.append(f"- {a['app']} — {a['minutes']:.0f} min")

    if entities:
        lines.append("\n## Worked with")
        lines.append(", ".join(f"{e['name']}" for e in entities[:12]))

    if claims:
        lines.append("\n## Accomplishments")
        for c in claims[:8]:
            lines.append(f"- {c['text']}")

    return "\n".join(lines)


def coach_prompt(metrics, claims=None):
    parts = [
        "You are a supportive but honest productivity coach. Based on the user's "
        "screen-activity metrics for the day, write 3-5 concise sentences of feedback: "
        "what went well, where focus was fragmented, and one concrete suggestion for "
        "tomorrow. Be specific to the data; do not invent activities.\n",
        f"Active minutes: {metrics.get('active_minutes')}",
        f"Focus score: {metrics.get('focus_score')}/100 "
        f"(avg block {metrics.get('avg_event_seconds', 0) / 60:.1f} min, "
        f"{metrics.get('switches')} switches, {metrics.get('switches_per_hour')}/hr)",
        "Time by activity: " + ", ".join(
            f"{a['activity']} {a['minutes']:.0f}m" for a in metrics.get("by_activity", [])),
    ]
    if metrics.get("by_project"):
        parts.append("Projects: " + ", ".join(
            f"{p['project']} {p['minutes']:.0f}m" for p in metrics["by_project"]))
    if claims:
        parts.append("Notable accomplishments: " + "; ".join(c["text"] for c in claims[:6]))
    return "\n".join(parts)

"""Phase 3 — the Coach: a productivity report + feedback.

`format_report` renders deterministic metrics (time per activity/app/project,
focus score, switches) into markdown, for a single day or for a longer period.
`coach_prompt` turns those metrics plus the day's accomplishments (claims) into a
prompt for the LLM to write a short, specific feedback note. The endpoint combines
both and posts the result into the Daily room as a 'coach' message.

Every number here is screen activity. Camera observations are excluded upstream by
the graph queries (neo4j_store.PRODUCTIVITY_DOMAIN) — the reports say so out loud,
because a report that silently changes what it counts is worse than one that
counted the wrong thing.
"""
from __future__ import annotations

from memory.summary.reports import improved, window_label

SOURCE_NOTE = "_Screen activity only — camera observations are not counted._"


def _bar(minutes, total):
    if not total:
        return ""
    n = int(round(10 * minutes / total))
    return "#" * n + "-" * (10 - n)


def _minutes(value):
    """Minutes as text, promoted to hours once 'min' stops being readable."""
    value = float(value or 0.0)
    if value < 90:
        return f"{value:.0f} min"
    return f"{value / 60.0:.1f} h"


def _signed(value, unit=""):
    return f"{value:+.0f}{unit}" if abs(value) >= 1 else f"{value:+.1f}{unit}"


def format_report(metrics, claims=None, entities=None, heading=None,
                  comparison=None, series=None):
    """Markdown for one window of metrics.

    `heading` overrides the default daily title (a weekly/monthly report passes
    its own). `comparison` and `series`, when given, add the previous-period and
    per-day sections that the charts show graphically.
    """
    span = (f"{metrics.get('start_date')} → {metrics.get('end_date')}"
            if metrics.get("start_date") != metrics.get("end_date")
            else metrics.get("date") or metrics.get("start_date"))
    lines = [f"# {heading or 'Daily report'} — {span}", "", SOURCE_NOTE, ""]
    active = metrics.get("active_minutes", 0)
    lines.append(
        f"**{active:.0f} min active** · {metrics.get('sessions', 0)} sessions · "
        f"{metrics.get('events', 0)} events · focus score **{metrics.get('focus_score', 0)}/100**")
    lines.append(
        f"_avg block {metrics.get('avg_event_seconds', 0) / 60:.1f} min · "
        f"longest {metrics.get('longest_block_seconds', 0) / 60:.1f} min · "
        f"{metrics.get('switches', 0)} context switches "
        f"({metrics.get('switches_per_hour', 0)}/hr)_")
    if metrics.get("active_days") and metrics.get("start_date") != metrics.get("end_date"):
        lines.append(f"_{metrics['active_days']} day(s) with activity_")
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

    if series and len(series) > 1:
        lines.append("\n## Day by day")
        peak = max((d["total_minutes"] for d in series), default=0) or 1
        for day in series:
            top = max(day["activities"].items(), key=lambda kv: kv[1],
                      default=(None, 0))[0]
            label = f"  ({top})" if top else ""
            lines.append(
                f"- {day['date']} {_minutes(day['total_minutes']):>7}  "
                f"{_bar(day['total_minutes'], peak)}{label}")

    if comparison:
        lines.append(format_comparison(comparison))

    if entities:
        lines.append("\n## Worked with")
        lines.append(", ".join(f"{e['name']}" for e in entities[:12]))

    if claims:
        lines.append("\n## Accomplishments")
        for c in claims[:8]:
            lines.append(f"- {c['text']}")

    return "\n".join(lines)


def format_comparison(comparison):
    """The 'vs baseline' section: headline deltas, then activities.

    The heading names the baseline the way it was chosen — a single day as one
    date, not as '2026-07-29 → 2026-07-29', which read as a day compared with
    itself. An averaged baseline says so, and says what both sides are per.
    """
    baseline = comparison.get("baseline_label") or window_label(
        comparison.get("previous_start_date"), comparison.get("previous_end_date"))
    lines = [f"\n## Compared with {baseline}"]
    if comparison.get("averaged"):
        current = comparison.get("current_label")
        lines.append(
            f"\n_Both sides are per active day{f' — this window is {current}' if current else ''}._")
    if not comparison.get("comparable", True):
        lines.append(
            "\n_Nothing was captured in that period, so there is nothing to "
            "compare against._")
        return "\n".join(lines)
    labels = {"active_minutes": "Active minutes", "events": "Events",
              "focus_score": "Focus score", "avg_event_seconds": "Avg block (s)",
              "switches_per_hour": "Switches/hr", "switches": "Switches"}
    for key, label in labels.items():
        entry = comparison.get("metrics", {}).get(key)
        if entry is None:
            continue
        better = improved(entry)
        mark = "" if better is None else (" better" if better else " worse")
        pct = f" ({entry['pct']:+.0f}%)" if entry["pct"] is not None else ""
        lines.append(
            f"- {label}: {entry['current']:.0f} vs {entry['previous']:.0f} "
            f"→ {_signed(entry['delta'])}{pct}{mark}")

    movers = [a for a in comparison.get("by_activity", []) if a["delta"]]
    if movers:
        lines.append("\n### Activity changes")
        for a in movers[:8]:
            lines.append(
                f"- {a['activity']}: {a['current']:.0f} vs {a['previous']:.0f} min "
                f"→ {_signed(a['delta'], ' min')}")
    return "\n".join(lines)


def coach_prompt(metrics, claims=None, comparison=None, period="day"):
    parts = [
        "You are a supportive but honest productivity coach. Based on the user's "
        f"screen-activity metrics for the {period}, write 3-5 concise sentences of "
        "feedback: what went well, where focus was fragmented, and one concrete "
        "suggestion for next time. Be specific to the data; do not invent "
        "activities. These metrics cover screen activity only — home camera "
        "observations are deliberately excluded, so do not treat their absence as "
        "idle time.\n",
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
    if comparison and comparison.get("comparable", True):
        against = comparison.get("baseline_label") or "the previous period"
        parts.append(f"Change vs {against}"
                     f"{' (both per active day)' if comparison.get('averaged') else ''}: "
                     + ", ".join(
            f"{key} {entry['delta']:+.0f}"
            for key, entry in comparison.get("metrics", {}).items() if entry["delta"]))
    if claims:
        parts.append("Notable accomplishments: " + "; ".join(c["text"] for c in claims[:6]))
    return "\n".join(parts)

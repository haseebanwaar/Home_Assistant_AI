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


def coach_prompt(metrics, claims=None, comparison=None, period="day",
                 reflection_context=""):
    """Feedback on the numbers, informed by what the user says about his life.

    `reflection_context` is his own Daily Reflection answers. Metrics say a day
    was fragmented; only he can say he was up with a sick relative, and a coach
    that ignores what he already wrote is worse than no coach.
    """
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
    if reflection_context:
        parts.append("\n" + reflection_context
                     + "\nWhere his own words explain the numbers, use them "
                       "rather than guessing at a cause, and hold the "
                       "suggestion to what he said he is trying to do.")
    return "\n".join(parts)


def _history_block(history):
    """The last fortnight of written reports, as calibration material.

    Scores only mean something as a series. Without the history a model rescores
    from scratch every night against an imagined ideal day, and the number walks
    — the same day scores 62 on Monday and 78 on Thursday. With it, today is
    scored against the days either side of it, which is the only comparison the
    user can actually act on.
    """
    if not history:
        return []
    lines = [
        "",
        "## Your previous reports",
        "These are the reports you wrote for the days before this one, newest "
        "last. They are here so this report continues them: score on the same "
        "dimensions where they still apply, calibrate today's numbers against "
        "these rather than against an ideal day, and say plainly when something "
        "you flagged before has or has not changed.",
    ]
    for entry in history:
        body = entry.get("report") or {}
        overall = entry.get("overall_score")
        if overall is None:
            overall = body.get("overall_score")
        head = entry.get("headline") or body.get("headline") or "(no headline)"
        lines.append("")
        lines.append(f"### {entry.get('end_date')} — overall "
                     f"{overall if overall is not None else 'not scored'}/100")
        lines.append(f"Headline: {head}")
        scored = [f"{s.get('name')} {s.get('score')}"
                  for s in (body.get("scores") or []) if s.get("name")]
        if scored:
            lines.append("Scores: " + ", ".join(scored))
        if body.get("next_step"):
            lines.append(f"You told him to: {body['next_step']}")
        for finding in (body.get("watchouts") or [])[:3]:
            if finding.get("text"):
                lines.append(f"You flagged: {finding['text']}")
    lines.append("")
    lines.append(
        "A score that cannot be read against those is not worth plotting. If "
        "you change what a dimension means, or drop one, say so in "
        "score_basis — an unexplained jump reads as the day changing when it "
        "was only the judge changing.")
    return lines


def report_prompt(metrics, claims=None, entities=None, comparison=None,
                  series=None, period="day", claim_summary=None,
                  reflection_context="", history=None, hours=None,
                  raw_report=""):
    """The brief for a written report, as opposed to a rendered one.

    `format_report` restates metrics; this hands the whole picture over and asks
    for judgement. Three things are deliberate here.

    *Everything is included.* Not the top eight apps but all of them, not a
    summary of the days but each day's split, not the surviving claims but the
    count that was dropped and why. A model asked to interpret a period with a
    truncated view of it will confidently interpret the truncation.

    *The shape is the writer's.* The old brief prescribed sections, forbade
    scoring, and fixed the label taxonomy. That produced reports that were
    correct and useless. What remains prescribed is honesty about evidence —
    everything else, including whether a section exists at all, is decided by
    whoever is writing.

    *The history is in the room.* `history` is the last fortnight of its own
    reports with their scores, so today is written as the next entry in a series
    rather than as a first impression.
    """
    lines = [
        "You are writing the activity report for the person whose screen time "
        "this is. He reads it to understand his own day. Write as someone who "
        "has been watching this record for weeks and has something to say about "
        "it — not as a dashboard, and not as a summarizer.",
        "",
        "## What is fixed",
        "Only this, and it is about evidence rather than about form:",
        "- Every claim rests on the data below or on his own words. Never "
        "invent an activity, an outcome, or an intention.",
        "- Screen activity only. Camera observations are deliberately excluded, "
        "so an absence here is not evidence that he was idle.",
        "- Time in an application is not achievement. A large number next to a "
        "remote-desktop, browser or file-manager window means that window was "
        "in the foreground, and nothing more.",
        "- 'Accomplishments' means things finished or genuinely moved forward. "
        "An empty list is the right answer to a day with no evidence of "
        "outcomes; say so in data_quality rather than padding.",
        "- Say when you think a number is wrong. The metrics are mechanical and "
        "you can see more than they can.",
        "",
        "## What is yours",
        "- The structure. `sections` takes any section this period needs, "
        "titled by you. Use the named fields when they fit and leave them empty "
        "when they do not.",
        "- The length. A thin day gets a short report.",
        "- The labels. The pipeline folds obvious spelling variants; it cannot "
        "tell that two differently-named projects are one effort, or that "
        "several executables are one activity. Where you count things together, "
        "count them together in your prose and record it in `label_merges` with "
        "the reason. Do not invent a category system he did not use — merge "
        "what is the same, in his own vocabulary, and leave the rest apart.",
        "- The judgement. Score the period on the dimensions you think matter "
        "for it (`scores`), plus one `overall_score` so the days can be "
        "plotted against each other. Scoring the day is not the same as "
        "judging the person: score the shape of the time and what came of it, "
        "with the rationale attached, and never diagnose him.",
        "- Disagreement. If the deterministic report below reads the period "
        "wrongly, say so and explain what you think actually happened.",
        "",
        "## The period",
        f"{period} ({metrics.get('start_date')} to {metrics.get('end_date')})",
        f"Active: {metrics.get('active_minutes', 0):.0f} min across "
        f"{metrics.get('active_days') or 1} active day(s), "
        f"{metrics.get('sessions', 0)} sessions, {metrics.get('events', 0)} events",
        f"Focus score {metrics.get('focus_score', 0)}/100 "
        f"(avg block {metrics.get('avg_event_seconds', 0) / 60:.1f} min, "
        f"longest {metrics.get('longest_block_seconds', 0) / 60:.0f} min, "
        f"{metrics.get('switches', 0)} switches at "
        f"{metrics.get('switches_per_hour', 0)}/hr)",
        "Time is only counted while he was actually at the machine — five "
        "minutes without keyboard or mouse and the clock stops — so these "
        "minutes are attended minutes, not minutes an app was open.",
    ]

    if metrics.get("by_activity"):
        lines.append("Time by activity: " + ", ".join(
            f"{a['activity']} {a['minutes']:.0f}m"
            for a in metrics["by_activity"]))

    if metrics.get("by_app"):
        lines.append("")
        lines.append("Time by application — every row, not a top slice. Names "
                     "are canonical; the note explains opaque executables:")
        for app in metrics["by_app"]:
            note = f" — {app['note']}" if app.get("note") else ""
            merged = (f" [already merged from {', '.join(app['aliases'])}]"
                      if len(app.get("aliases") or []) > 1 else "")
            lines.append(
                f"  - {app.get('label', app.get('app'))} "
                f"({app.get('category', 'other')}): {app['minutes']:.0f}m"
                f"{merged}{note}")

    if metrics.get("by_project"):
        lines.append("")
        lines.append("Time by project (spelling variants already merged; "
                     "genuine duplicates across names are yours to merge):")
        for project in metrics["by_project"]:
            merged = (f" [already merged from {', '.join(project['aliases'])}]"
                      if len(project.get("aliases") or []) > 1 else "")
            lines.append(
                f"  - {project.get('label', project.get('project'))}: "
                f"{project['minutes']:.0f}m{merged}")

    if series and len(series) > 1:
        lines.append("")
        lines.append("Day by day, with each day's full split:")
        for day in series:
            split = ", ".join(
                f"{name} {value:.0f}m" for name, value in
                sorted(day.get("activities", {}).items(),
                       key=lambda kv: -kv[1]))
            lines.append(
                f"  - {day['date']}: {day['total_minutes']:.0f}m"
                + (f" — {split}" if split else " — nothing recorded"))

    if hours and any(hours):
        lines.append("")
        lines.append("When the time fell, by hour of his local day:")
        lines.append("  " + ", ".join(
            f"{hour:02d}:00 {minutes:.0f}m"
            for hour, minutes in enumerate(hours) if minutes >= 1))

    if comparison and comparison.get("comparable", True):
        against = comparison.get("baseline_label") or "the previous period"
        lines.append("")
        lines.append(f"Against {against}"
                     + (" (both sides per active day)"
                        if comparison.get("averaged") else "") + ":")
        for key, entry in (comparison.get("metrics") or {}).items():
            lines.append(
                f"  - {key}: {entry.get('current')} vs {entry.get('previous')} "
                f"({entry.get('delta'):+})")
        for row in (comparison.get("by_activity") or [])[:12]:
            lines.append(
                f"  - {row.get('activity')}: {row.get('current')}m vs "
                f"{row.get('previous')}m ({row.get('delta'):+}m)")
    elif comparison:
        lines.append("")
        lines.append(
            "The baseline period captured nothing, so there is no comparison "
            "to draw. Do not present the current numbers as an improvement.")

    if entities:
        lines.append("")
        lines.append("Entities seen: " + ", ".join(
            e["name"] for e in entities))

    if claims:
        lines.append("")
        lines.append(
            "Candidate accomplishments, already filtered for informativeness "
            "and ranked (score 0-1). Treat these as raw evidence, not as "
            "finished report lines — merge, rewrite, and discard freely:")
        for claim in claims:
            lines.append(f"  - [{claim.get('score', 0):.2f}] {claim['text']}")
    else:
        lines.append("")
        lines.append(
            "No candidate accomplishment survived filtering. Every extracted "
            "claim was a bare observation of looking at a screen. Say that "
            "plainly in data_quality and return an empty accomplishments list.")

    if claim_summary and claim_summary.get("dropped"):
        lines.append(
            f"({claim_summary['dropped']} of {claim_summary['considered']} "
            "extracted claims were dropped as uninformative.)")

    if raw_report:
        lines.append("")
        lines.append("## The deterministic report")
        lines.append(
            "This is what the app renders from the same numbers, shown to him "
            "beside yours. Yours is not a rewrite of it — do not restate it, "
            "and contradict it where you think it is wrong:")
        lines.append(raw_report)

    lines.extend(_history_block(history))

    if reflection_context:
        lines.append("")
        lines.append(reflection_context)
        lines.append(
            "These are the only statements here that come from the user "
            "himself. They outrank every inference in this prompt, including "
            "yours. Use them for intent and context — what the period was for, "
            "what he was trying to do — but never as evidence that something "
            "was done. Activity data alone establishes that.")

    return "\n".join(lines)

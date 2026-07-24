"""Step 13 — daily summarizer + Obsidian export (plan §11, §16).

Builds a deterministic daily note (timeline -> sessions -> events, top entities,
notable claims) and a Pending-Merges note (from POSSIBLY_SAME_AS candidates),
then writes them into an Obsidian vault. Entities are linked as [[wikilinks]] so
the vault becomes a navigable knowledge graph. No VLM needed — reads the graph.
"""
from __future__ import annotations

import datetime
import os
import re


def _hm(ts):
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
    except (OverflowError, OSError, ValueError, TypeError):
        return "--:--"


def _mins(seconds):
    return f"{(seconds or 0) / 60:.0f} min"


def _wikilink(name):
    """Obsidian-safe [[wikilink]] — strip characters that break link syntax."""
    safe = re.sub(r"[\[\]|#^]", "", str(name or "")).strip()
    return f"[[{safe}]]" if safe else ""


def build_daily_note(store, date_str):
    sessions = store.sessions_with_events(date_str)
    # entities per event (one query for the whole day)
    event_ids = [ev["event_id"] for s in sessions for ev in (s.get("events") or [])
                 if ev and ev.get("event_id")]
    ents_by_event = store.entities_for_events(event_ids) if event_ids else {}
    top_entities = store.day_entities(date_str)
    claims = store.day_claims(date_str)

    total_active = sum(s.get("active_seconds") or 0 for s in sessions)

    lines = [f"# {date_str}", ""]
    lines.append(f"**Active:** {_mins(total_active)} across {len(sessions)} session(s).")
    lines.append("")

    lines.append("## Timeline")
    if not sessions:
        lines.append("_No sessions recorded._")
    for s in sessions:
        proj = f" · {s['project_id']}" if s.get("project_id") else ""
        header = (f"### {s.get('application', '?')} — {s.get('activity', '?')}"
                  f"{proj}  ({_mins(s.get('active_seconds'))}"
                  f"{', ' + str(s['resume_count']) + ' resume(s)' if s.get('resume_count') else ''})")
        lines.append(header)
        for ev in (s.get("events") or []):
            if not ev:
                continue
            when = f"{_hm(ev.get('span_start'))}–{_hm(ev.get('span_end'))}"
            summary = (ev.get("summary") or "").strip().replace("\n", " ")
            ents = ents_by_event.get(ev.get("event_id"), [])
            # link the most salient entities (primary first), cap the list
            names = [e["name"] for e in sorted(ents, key=lambda x: x.get("role") != "primary")][:6]
            links = "  ".join(_wikilink(n) for n in names if n)
            lines.append(f"- **{when}** — {summary[:160]}")
            if links:
                lines.append(f"    - {links}")
        lines.append("")

    lines.append("## Top entities")
    for e in top_entities:
        lines.append(f"- {_wikilink(e['name'])} ({e.get('type', 'other')}) ×{e['mentions']}")
    if not top_entities:
        lines.append("_None._")
    lines.append("")

    if claims:
        lines.append("## Notable claims")
        for c in claims:
            conf = c.get("confidence")
            tag = f" _(conf {conf:.2f})_" if isinstance(conf, (int, float)) else ""
            lines.append(f"- {c['text']}{tag}")
        lines.append("")

    lines.append("## See also")
    lines.append("- [[Pending-Merges]]")
    lines.append("")
    return "\n".join(lines)


def build_pending_merges(store):
    cands = store.possibly_same_as()
    lines = ["# Pending-Merges", "",
             "Entity alias candidates awaiting review. Each pair is a *soft* "
             "`POSSIBLY_SAME_AS` suggestion — nothing has been merged.", ""]
    if not cands:
        lines.append("_No pending candidates._")
        return "\n".join(lines) + "\n"
    lines.append("| A | B | score | method |")
    lines.append("| --- | --- | --- | --- |")
    for c in cands:
        lines.append(f"| {_wikilink(c['a'])} | {_wikilink(c['b'])} | "
                     f"{c.get('score')} | {c.get('method')} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def export_to_vault(store, vault_dir, date_str=None):
    """Write the daily note + Pending-Merges into the Obsidian vault.

    Returns the paths written.
    """
    date_str = date_str or datetime.date.today().isoformat()
    daily_dir = os.path.join(vault_dir, "Daily")
    os.makedirs(daily_dir, exist_ok=True)

    daily_path = os.path.join(daily_dir, f"{date_str}.md")
    with open(daily_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_daily_note(store, date_str))

    merges_path = os.path.join(vault_dir, "Pending-Merges.md")
    with open(merges_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_pending_merges(store))

    return {"daily_note": daily_path, "pending_merges": merges_path}

"""Shared batch-context derivation (Step-a refactor).

Given a batch/observation carrying `window_titles` and `process_names`, derive
the representative application, window title, domain profile, and project id.
Used by BOTH the offline session tool and the live memory pipeline so routing
never drifts between them.

Routing uses the batch's REPRESENTATIVE (final) app — the last process/title —
not the union of everything seen that minute, so a browser batch that briefly
touched an IDE is not mis-routed to coding.
"""
from __future__ import annotations

import re

from memory.profiles.registry import select_profile

_DASH = r"[–—―�\-]"


def app_of(obs):
    """Representative app: last process name, else derived from the title suffix."""
    procs = obs.get("process_names") or []
    if procs:
        return procs[-1].lower()
    titles = obs.get("window_titles") or []
    if titles:
        t = titles[-1]
        return (t.rsplit(" - ", 1)[-1] if " - " in t else t).strip().lower()
    return ""


def title_of(obs):
    titles = obs.get("window_titles") or []
    return (titles[-1] if titles else "").strip()


def profile_name(obs):
    procs = obs.get("process_names") or []
    titles = obs.get("window_titles") or []
    return select_profile(procs[-1:], titles[-1:]).name


def normalize_name(name):
    """Canonical key for exact/alias entity matching (Step 9)."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def project_of(obs):
    """Coding batches get a project id parsed from the IDE title; else None."""
    if profile_name(obs) != "coding":
        return None
    title = title_of(obs)
    # IDE titles look like "<project> <dash> <path/file>"; take the first segment.
    first = re.split(rf"\s{_DASH}\s", title, maxsplit=1)[0].strip()
    return (first or (title.split()[0] if title else None)) or None

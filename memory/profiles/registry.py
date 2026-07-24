"""Step 3 — domain profiles (plan §8.1).

A profile supplies (a) a domain entity vocabulary and (b) prompt guidance, so the
VLM stops emitting vague entities like "code"/"screen" and instead names files,
functions, and libraries. Profiles are selected per observation by process_name
(preferred), then by a window-title keyword heuristic (so replays of captures
that lack process_name still route), and finally fall back to generic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Profile:
    name: str
    # Executable names that route to this profile (lowercased, incl. ".exe").
    match_processes: frozenset = field(default_factory=frozenset)
    # Substrings in a window title that route to this profile (lowercased).
    match_title_keywords: tuple = ()
    # Suggested entity types for this domain (fed into the prompt).
    entity_types: tuple = ()
    # Domain guidance appended to the extraction system prompt.
    focus: str = ""


def _norm(values):
    return [v.lower() for v in values if v]


def select_profile(process_names=None, window_titles=None):
    """Pick the best-matching profile.

    process_names / window_titles are lists (a batch may span several windows).
    process_name wins; then title keywords; else generic.
    """
    from memory.profiles.coding import CODING_PROFILE
    from memory.profiles.generic import GENERIC_PROFILE

    profiles = [CODING_PROFILE]  # order = priority; extend as domains are added

    procs = _norm(process_names or [])
    titles = _norm(window_titles or [])

    for p in profiles:
        if any(proc in p.match_processes for proc in procs):
            return p
    for p in profiles:
        if any(kw in title for title in titles for kw in p.match_title_keywords):
            return p
    return GENERIC_PROFILE

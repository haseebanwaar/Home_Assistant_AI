"""Step 12 — entity resolution (plan §10).

Proposes POSSIBLY_SAME_AS links between entities that are likely the same, WITHOUT
merging them (no hard merges — that's a later consolidation decision). Two
methods, both high-precision:

1. canonical — strip all non-alphanumerics and lowercase; equal canonical forms
   are near-certain aliases ("Qwen3-VL" == "Qwen 3 VL" == "qwen3vl"). score 1.0.
2. fuzzy — SequenceMatcher ratio over canonical forms, but ONLY when the two
   names share the same set of digit-runs and are type-compatible. The digit
   guard is what stops version-numbered files ("...20270206.py" vs
   "...20220726.py") from being wrongly linked. score = ratio.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from itertools import combinations


def canonical(name):
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _digits(name):
    return tuple(re.findall(r"\d+", name or ""))


def _type_compatible(t1, t2):
    if not t1 or not t2:
        return True
    if t1 == t2:
        return True
    return t1 == "other" or t2 == "other"


def find_same_as_candidates(entities, fuzzy_threshold=0.9):
    """entities: list of {entity_id, name, type}. Returns candidate pair dicts
    {a, b, score, method} with a < b by entity_id (stable, no duplicate edges)."""
    ents = [e for e in entities if canonical(e.get("name"))]
    pairs = []
    seen = set()

    # 1) Canonical-form equality (high precision).
    by_canon = {}
    for e in ents:
        by_canon.setdefault(canonical(e["name"]), []).append(e)
    for group in by_canon.values():
        if len(group) < 2:
            continue
        for a, b in combinations(group, 2):
            if a["entity_id"] == b["entity_id"]:
                continue
            key = tuple(sorted((a["entity_id"], b["entity_id"])))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"a": key[0], "b": key[1], "score": 1.0, "method": "canonical"})

    # 2) Fuzzy — same digits, type-compatible, high ratio.
    for a, b in combinations(ents, 2):
        if a["entity_id"] == b["entity_id"]:
            continue
        key = tuple(sorted((a["entity_id"], b["entity_id"])))
        if key in seen:
            continue
        if _digits(a["name"]) != _digits(b["name"]):
            continue
        if not _type_compatible(a.get("type"), b.get("type")):
            continue
        ratio = SequenceMatcher(None, canonical(a["name"]), canonical(b["name"])).ratio()
        if ratio >= fuzzy_threshold:
            seen.add(key)
            pairs.append({"a": key[0], "b": key[1], "score": round(ratio, 3), "method": "fuzzy"})

    return pairs

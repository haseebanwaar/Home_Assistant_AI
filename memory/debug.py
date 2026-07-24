"""Append-only JSONL debug sinks under data/debug/.

Every pipeline stage gets a debug handle here (observations.jsonl in Step 1,
extractions.jsonl in Step 2, boundaries.jsonl in Step 5, ...). One record per
line so `tail -f` and line-by-line replay both work.
"""
import json
import os
import threading

DEBUG_DIR = os.getenv("DEBUG_DIR", os.path.join("data", "debug"))

_lock = threading.Lock()


def write_jsonl(name, record):
    """Append one record to data/debug/<name>.jsonl.

    `record` may be a pydantic model (has .model_dump) or a plain dict.
    """
    if hasattr(record, "model_dump"):
        record = record.model_dump()
    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, f"{name}.jsonl")
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
    return path

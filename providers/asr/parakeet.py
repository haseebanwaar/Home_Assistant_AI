"""Lightweight client for the persistent Parakeet inference server."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


def _server_url() -> str:
    # Resolve this at request time because app.py loads .env after importing us.
    return os.getenv("PARAKEET_SERVER_URL", "http://127.0.0.1:8765").rstrip("/")


def parakeet_health(timeout: float = 1.0) -> dict:
    """Return the persistent server health without raising on connection errors."""
    try:
        with urlopen(f"{_server_url()}/health", timeout=timeout) as response:
            payload = json.load(response)
        return {
            "connected": True,
            "ready": payload.get("ready") is True,
            "model_path": payload.get("model_path"),
        }
    except Exception as exc:
        return {"connected": False, "ready": False, "error": str(exc)}


def nemo_transcribe(data) -> str:
    """Send 16 kHz float32 mono samples to the persistent ASR process."""
    audio = np.ascontiguousarray(data, dtype=np.float32)
    request = Request(
        f"{_server_url()}/transcribe",
        data=audio.tobytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": "16000",
        },
        method="POST",
    )
    timeout = float(os.getenv("PARAKEET_SERVER_TIMEOUT", "120"))

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Parakeet server returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Parakeet server is unavailable at {_server_url()}; "
            "start providers/asr/parakeet_server.py first"
        ) from exc

    return str(payload.get("text", "")).strip()

"""Persistent Parakeet/NeMo inference service for fast backend debugging.

Run this file in a separate terminal and leave it running while restarting
``app.py``. The expensive model is loaded exactly once in this process.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from omegaconf import open_dict

import nemo.collections.asr as nemo_asr

load_dotenv()

logger = logging.getLogger("parakeet_server")

MODEL_PATH = os.getenv(
    "PARAKEET_MODEL_PATH",
    r"d:\models\tts\gguf\parakeet-tdt-0.6b-v3.nemo",
)
MODEL_NAME = os.getenv("PARAKEET_MODEL_NAME", "nvidia/parakeet_ctc_small")
HOST = os.getenv("PARAKEET_SERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("PARAKEET_SERVER_PORT", "8765"))

_model = None
_inference_lock = asyncio.Lock()


def _load_model() -> None:
    global _model
    logger.info("Loading Parakeet model; this is the one slow startup...")
    try:
        _model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
        logger.info("Loaded local model from %s", MODEL_PATH)
    except Exception as exc:
        logger.warning("Could not load %s (%s); loading %s", MODEL_PATH, exc, MODEL_NAME)
        _model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)

    decoding_cfg = _model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.greedy.use_cuda_graph_decoder = False
    _model.change_decoding_strategy(decoding_cfg)


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _transcribe(audio: np.ndarray) -> dict[str, Any]:
    output = _model.transcribe([audio], timestamps=True)
    if not output:
        return {"text": "", "word_timestamps": []}

    hypothesis = output[0]
    timestamps = getattr(hypothesis, "timestamp", None) or {}
    words = timestamps.get("word", [])
    text = (getattr(hypothesis, "text", None) or "").strip()
    if not text and words:
        text = " ".join(
            str(word.get("word", "")).strip()
            for word in words
            if str(word.get("word", "")).strip()
        )
    if not text and isinstance(hypothesis, str):
        text = hypothesis.strip()
    return {"text": text, "word_timestamps": _json_safe(words)}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(_load_model)
    yield


app = FastAPI(title="Persistent Parakeet ASR", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ready": _model is not None, "model_path": MODEL_PATH}


@app.post("/transcribe")
async def transcribe(request: Request):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    if request.headers.get("x-sample-rate", "16000") != "16000":
        raise HTTPException(status_code=400, detail="Only 16 kHz audio is supported")

    body = await request.body()
    if len(body) % np.dtype(np.float32).itemsize:
        raise HTTPException(status_code=400, detail="Body must contain raw float32 samples")
    audio = np.frombuffer(body, dtype=np.float32)
    if len(audio) < 3200:
        return {"text": "", "word_timestamps": []}

    # NeMo/CUDA inference is not safe to overlap on this model instance.
    async with _inference_lock:
        try:
            return await asyncio.to_thread(_transcribe, audio)
        except Exception as exc:
            logger.exception("Transcription failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

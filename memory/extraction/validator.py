"""Parse + validate the VLM's JSON into an ExtractionResult (Step 2).

Contract: parse -> validate -> one retry with the error fed back -> on 2nd
failure fall back to a minimal ExtractionResult({summary: <prose>, entities: []})
so the pipeline never crashes on a bad generation.
"""
import json
import logging

from pydantic import ValidationError

from memory.models.extraction import ExtractionResult
from memory.extraction.prompts import RETRY_PREFIX, RETRY_SUFFIX

logger = logging.getLogger("home_assistant")


def _strip_fences(text):
    """Remove ```json ... ``` fences and grab the outermost JSON object."""
    t = (text or "").strip()
    if t.startswith("```"):
        # drop leading ```lang and trailing ```
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    t = t.strip()
    # If there's leading/trailing prose, keep the outermost {...}.
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start : end + 1]
    return t


def parse_extraction(raw_text):
    """Parse raw VLM text into an ExtractionResult. Raises on failure."""
    cleaned = _strip_fences(raw_text)
    data = json.loads(cleaned)
    return ExtractionResult.model_validate(data)


def _fallback(raw_text):
    """Never-crash fallback: keep the prose as summary, no entities."""
    summary = (raw_text or "").strip() or "No description available."
    return ExtractionResult(summary=summary, confidence=0.1, entities=[])


async def run_extraction(generate):
    """Drive one extraction with a single validating retry.

    `generate(feedback)` is an async callable returning raw VLM text; when
    `feedback` is not None it should append that correction instruction to the
    prompt. Returns (ExtractionResult, status) where status is one of
    "ok" | "retry" | "fallback".
    """
    raw = await generate(None)
    try:
        return parse_extraction(raw), "ok"
    except (ValidationError, ValueError, json.JSONDecodeError) as first_err:
        logger.warning("Extraction failed validation, retrying once: %s", first_err)
        feedback = f"{RETRY_PREFIX}{first_err}{RETRY_SUFFIX}"
        try:
            raw2 = await generate(feedback)
        except Exception as gen_err:
            logger.warning("Retry generation failed: %s", gen_err)
            return _fallback(raw), "fallback"
        try:
            return parse_extraction(raw2), "retry"
        except (ValidationError, ValueError, json.JSONDecodeError) as second_err:
            logger.warning("Extraction failed again, falling back to prose: %s", second_err)
            return _fallback(raw2 or raw), "fallback"

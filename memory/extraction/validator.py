"""Parse + validate the VLM's JSON into an ExtractionResult (Step 2).

Contract: parse -> validate -> one retry with the error fed back -> on 2nd
failure fall back to a minimal ExtractionResult({summary: <prose>, entities: []})
so the pipeline never crashes on a bad generation.
"""
import json
import logging

from pydantic import ValidationError

from memory.models.extraction import (
    Claim, Entity, ExtractionResult, PersonalMemoryCandidate, SceneState,
)
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


def _salvage_fields(data):
    """Best-effort ExtractionResult from a dict that failed whole-object validation.

    A response is usually rejected over ONE bad field, so re-validating field by
    field recovers the summary/entities/claims instead of discarding everything.
    Returns None when there is no usable summary.
    """
    summary = data.get("summary")
    summary = str(summary).strip() if summary is not None else ""
    if not summary:
        return None

    entities = []
    for raw in data.get("entities") or []:
        try:
            entities.append(Entity.model_validate(raw))
        except (ValidationError, TypeError):
            continue
    claims = []
    for raw in data.get("claims") or []:
        try:
            claims.append(Claim.model_validate(raw))
        except (ValidationError, TypeError):
            continue
    personal = []
    for raw in data.get("personal_memory") or []:
        try:
            personal.append(PersonalMemoryCandidate.model_validate(raw))
        except (ValidationError, TypeError):
            continue
    # Salvaged states matter as much as salvaged claims: dropping them makes the
    # camera look at an unchanged scene and conclude the tracked things all left.
    states = []
    for raw in data.get("states") or []:
        try:
            states.append(SceneState.model_validate(raw))
        except (ValidationError, TypeError):
            continue
    gone = [str(key).strip() for key in (data.get("gone") or [])
            if isinstance(key, str) and str(key).strip()]

    # Add the optional fields one at a time, re-validating the whole model each
    # time, so a single bad value falls back to its default instead of being
    # written straight through (plain setattr skips pydantic validation).
    base = {"summary": summary, "confidence": 0.3,
            "entities": entities, "claims": claims,
            "personal_memory": personal, "states": states, "gone": gone}
    result = ExtractionResult(**base)
    for field in ("activity_type", "event_type", "project", "importance"):
        if field not in data:
            continue
        try:
            candidate = ExtractionResult(**{**base, field: data[field]})
        except (ValidationError, TypeError):
            continue
        base[field] = data[field]
        result = candidate
    return result


def _fallback(raw_text):
    """Never-crash fallback.

    Crucially this must NEVER store a raw JSON blob as the summary: the summary
    is what gets embedded, retrieved, cited and spoken, so a leaked blob poisons
    every downstream surface. When the response IS json-ish we salvage its real
    fields; when it is prose we keep the prose; otherwise we store a placeholder.
    """
    text = (raw_text or "").strip()
    if not text:
        return ExtractionResult(summary="No description available.",
                                confidence=0.1, entities=[])

    cleaned = _strip_fences(text)
    data = None
    try:
        data = json.loads(cleaned)
    except (ValueError, json.JSONDecodeError):
        pass

    if isinstance(data, dict):
        salvaged = _salvage_fields(data)
        if salvaged is not None:
            return salvaged
        # Structured but unusable — storing it verbatim is worse than nothing.
        logger.warning("Extraction fallback: JSON object without a usable summary; "
                       "discarding the blob.")
        return ExtractionResult(summary="No description available.",
                                confidence=0.1, entities=[])

    # Unparseable but still JSON-shaped (e.g. truncated mid-object): not prose.
    if cleaned.startswith("{") or cleaned.startswith("["):
        logger.warning("Extraction fallback: malformed JSON, discarding the blob.")
        return ExtractionResult(summary="No description available.",
                                confidence=0.1, entities=[])

    return ExtractionResult(summary=text, confidence=0.1, entities=[])


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

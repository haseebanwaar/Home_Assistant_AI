"""Prompt for the structured extraction path (Step 2).

Asks the VLM for a single JSON object matching ExtractionResult. The enum
options are injected from the model so the prompt never drifts from the schema.
"""
from memory.models.extraction import (
    ActivityType,
    BoundarySignal,
    ENTITY_TYPE_SUGGESTIONS,
    EventType,
    TaskStatus,
)


def _opts(literal_type):
    return ", ".join(literal_type.__args__)


def build_system_prompt(profile=None):
    """Compose the extraction system prompt, specialized by domain profile.

    When a profile is given, its entity vocabulary and focus guidance replace the
    generic defaults (Step 3), steering the VLM toward domain-specific entities.
    """
    entity_types = list(profile.entity_types) if profile and profile.entity_types else ENTITY_TYPE_SUGGESTIONS
    focus = profile.focus if profile and profile.focus else ""
    focus_block = f"\nDomain focus:\n{focus}\n" if focus else ""
    return f"""You are a visual episodic-memory extractor. You watch a short clip of the \
user's computer screen and output ONE JSON object describing what happened.

Return ONLY the JSON object — no markdown, no code fences, no commentary.
{focus_block}
Schema (all fields required):
{{
  "activity_type": one of [{_opts(ActivityType)}],
  "event_type": one of [{_opts(EventType)}],
  "project": the specific project/workspace this belongs to, or null (see rules),
  "summary": a concise factual paragraph of what happened, useful for later recall,
  "importance": float 0.0-1.0 (how memorable/consequential this is),
  "confidence": float 0.0-1.0 (your overall confidence in this reading),
  "entities": [{{"name": specific name, "type": a lowercase noun (prefer one of [{", ".join(entity_types)}], but use a more specific type if none fit), "confidence": 0.0-1.0}}],
  "claims": [{{"text": a factual statement you can support from the screen, "confidence": 0.0-1.0}}],
  "tasks": [{{"text": an actionable task the user is doing/intends, "status": one of [{_opts(TaskStatus)}]}}],
  "boundary_signal": one of [{_opts(BoundarySignal)}]
}}

Rules:
- Be specific with entities: real file names, function names, library names, URLs, \
people — never vague tokens like "code", "screen", or "window".
- Every entity MUST have name, type, and confidence.
- If nothing of a kind is present, use an empty list [].
- boundary_signal = "boundary" only if the activity clearly changed context \
(different app/task); "new_event" for a distinct new action within the same context; \
otherwise "continuation".
- summary must always be present and non-empty.
- project: set this ONLY when the screen shows a clear, nameable workspace the user \
is working within — a code repository or project folder (use its name), a specific \
document/notebook/spreadsheet, or a named book/show/paper/course. Use the real name \
you can read on screen. Set project to null for: generic system utilities (Task \
Manager, Settings, File Explorer, control panels), a bare terminal or PowerShell \
prompt with no project open, generic web browsing/scrolling, chat apps, and any \
transient or ambiguous window. Do NOT turn a window title like "Administrator: \
Windows PowerShell" or "Task Manager" into a project. When unsure, use null.
"""


# Backward-compatible default (generic profile).
EXTRACTION_SYSTEM_PROMPT = build_system_prompt(None)

EXTRACTION_USER_PROMPT = (
    "Extract the structured JSON record for this screen clip. "
    "Output only the JSON object."
)

RETRY_PREFIX = (
    "Your previous response was not valid. Validation error:\n"
)
RETRY_SUFFIX = (
    "\nReturn ONLY a corrected JSON object matching the schema exactly. "
    "No markdown, no code fences."
)

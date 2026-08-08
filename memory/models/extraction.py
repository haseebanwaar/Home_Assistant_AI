"""Step 2 — structured extraction contract (plan §8.3).

The VLM emits a validated ExtractionResult (JSON) instead of loose prose, but it
still carries a `summary` string so the existing Qdrant activity log keeps
working unchanged. Enums are strict so drift is caught by validation.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ActivityType = Literal[
    "coding", "browsing", "reading", "writing", "watching",
    "communication", "gaming", "terminal", "design", "other",
]

EventType = Literal[
    "start", "progress", "switch", "complete", "idle", "other",
]

# Entity type is OPEN-vocabulary: the world has more entity kinds (paper, model,
# dataset, subreddit, benchmark, ...) than any closed enum can hold, and the VLM
# reliably tags them with sensible-but-unlisted types. We suggest these in the
# prompt and normalize, but never reject an unknown type — that was the sole
# cause of extraction fallbacks. Strict enums stay on the fields that drive logic.
ENTITY_TYPE_SUGGESTIONS = [
    "person", "organization", "project", "file", "function", "class",
    "library", "framework", "tool", "application", "url", "website",
    "paper", "document", "model", "dataset", "benchmark", "repository",
    "topic", "concept", "product", "course", "article", "other",
]

BoundarySignal = Literal["continuation", "new_event", "boundary"]

TaskStatus = Literal["todo", "in_progress", "done", "blocked"]


class Entity(BaseModel):
    name: str
    type: str = "other"  # open vocabulary; see ENTITY_TYPE_SUGGESTIONS
    confidence: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, v):
        if not v or not isinstance(v, str):
            return "other"
        return v.strip().lower().replace(" ", "_")


class Claim(BaseModel):
    text: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class Task(BaseModel):
    text: str
    status: TaskStatus = "todo"


class SceneState(BaseModel):
    """A standing condition of something a fixed camera keeps watching.

    Unlike an Entity (something present in this window) or a Claim (something
    that happened in it), a SceneState is what is *still true* — "the black gate
    is closed", "the orange car is parked nose-out". `key` refers back to a slot
    the extractor was shown, which is how the same physical thing stays the same
    tracked thing across clips; it is null when the camera has not seen this
    before. See sources.camera_state.
    """
    key: Optional[str] = None
    subject: str
    state: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class PersonalMemoryCandidate(BaseModel):
    # Open vocabulary on purpose: personal memory should not be limited to a
    # developer's guess at which parts of a life may matter.
    category: str = "other"
    name: str
    value: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)


_NULLISH = {"", "none", "null", "n/a", "na", "unknown", "untitled"}


class ExtractionResult(BaseModel):
    activity_type: ActivityType = "other"
    event_type: EventType = "other"
    summary: str
    # The specific project/workspace this activity belongs to, decided by the VLM
    # (a code repo/folder, a named document/book/show/paper). null for generic
    # system tools, bare terminals, generic browsing — so no spurious room forms.
    project: Optional[str] = None
    importance: float = Field(0.5, ge=0.0, le=1.0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    entities: List[Entity] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    personal_memory: List[PersonalMemoryCandidate] = Field(default_factory=list)
    # Camera path only (the screen has no standing scene to carry between
    # windows): what is still true, and which tracked things have left.
    states: List[SceneState] = Field(default_factory=list)
    gone: List[str] = Field(default_factory=list)
    boundary_signal: BoundarySignal = "continuation"

    @model_validator(mode="before")
    @classmethod
    def _repair_common_vlm_drift(cls, value):
        """Repair harmless schema drift without another multimodal inference.

        Local VLMs commonly emit claims as strings and occasionally confuse the
        event and boundary enums. Both forms contain enough information to repair
        deterministically, so making the model inspect every image again is wasteful.
        """
        if not isinstance(value, dict):
            return value
        data = dict(value)

        claims = data.get("claims")
        if isinstance(claims, list):
            data["claims"] = [
                {"text": claim.strip(), "confidence": 0.5}
                if isinstance(claim, str) and claim.strip()
                else claim
                for claim in claims
                if not isinstance(claim, str) or claim.strip()
            ]

        # Scene states drive persistent camera tracking, where one malformed
        # entry must never cost the whole window: a dropped state is a state the
        # next clip re-confirms, but a failed extraction loses the event too. So
        # unusable entries are discarded here rather than raised.
        states = data.get("states")
        if isinstance(states, list):
            repaired = []
            for state in states:
                if isinstance(state, SceneState):
                    # Already validated (the salvage path builds these) — pass
                    # it through rather than mistaking it for junk.
                    repaired.append(state)
                    continue
                if not isinstance(state, dict):
                    continue
                subject = str(state.get("subject") or "").strip()
                text = str(state.get("state") or "").strip()
                if subject and text:
                    repaired.append({**state, "subject": subject, "state": text})
            data["states"] = repaired

        # "gone" is a list of slot keys, but models reach for the object form
        # they just used for states.
        gone = data.get("gone")
        if isinstance(gone, list):
            keys = []
            for item in gone:
                if isinstance(item, dict):
                    item = item.get("key") or item.get("subject") or ""
                item = str(item or "").strip()
                if item:
                    keys.append(item)
            data["gone"] = keys

        # Nothing downstream branches on event_type (unlike activity_type, which
        # routes rooms and so stays strict), so an unlisted value is never worth
        # a second multimodal pass: keep the boundary word when the model merely
        # swapped the two enums, and otherwise degrade to "other".
        event_type = data.get("event_type")
        if event_type is not None and event_type not in EventType.__args__:
            if (event_type in BoundarySignal.__args__
                    and data.get("boundary_signal") in (None, "", "continuation")):
                data["boundary_signal"] = event_type
            data["event_type"] = "other"
        return data

    @field_validator("project", mode="before")
    @classmethod
    def _clean_project(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        return None if s.lower() in _NULLISH else s

    @field_validator("boundary_signal", mode="before")
    @classmethod
    def _coerce_boundary_signal(cls, v):
        # Nothing downstream reads boundary_signal (the pipeline computes its own
        # boundary from visual/app change), yet the VLM sometimes puts an
        # event_type value like "idle" here. Coerce anything unrecognized to the
        # default instead of failing the whole extraction into a prose fallback.
        return v if v in BoundarySignal.__args__ else "continuation"

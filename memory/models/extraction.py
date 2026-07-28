"""Step 2 — structured extraction contract (plan §8.3).

The VLM emits a validated ExtractionResult (JSON) instead of loose prose, but it
still carries a `summary` string so the existing Qdrant activity log keeps
working unchanged. Enums are strict so drift is caught by validation.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

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
    boundary_signal: BoundarySignal = "continuation"

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

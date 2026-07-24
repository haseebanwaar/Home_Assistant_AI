"""Generic fallback profile (Step 3)."""
from memory.models.extraction import ENTITY_TYPE_SUGGESTIONS
from memory.profiles.registry import Profile

GENERIC_PROFILE = Profile(
    name="generic",
    match_processes=frozenset(),
    match_title_keywords=(),
    entity_types=tuple(ENTITY_TYPE_SUGGESTIONS),
    focus=(
        "Identify the specific entities on screen by their real names — people, "
        "organizations, papers, products, URLs, topics — never vague tokens."
    ),
)

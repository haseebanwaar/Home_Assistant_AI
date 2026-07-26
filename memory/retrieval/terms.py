"""Query tokenization for keyword search.

The graph search used to match the WHOLE question as one substring, so a
natural-language query ("what was I doing with the camera worker") could never
appear verbatim inside an event summary and always returned nothing. We split
the query into content terms instead and score by how many of them a record hits.

Stopwords are dropped so question scaffolding ("what", "did", "the") neither
inflates scores nor matches every row. A query made ONLY of stopwords — "what
did I do today" — yields no terms at all; that is deliberate. Such questions are
answered by scope (a date/room window), not by keywords, and the caller falls
back to a chronological fetch rather than to keyword noise.
"""
from __future__ import annotations

import re

# Question scaffolding + very common English function words. Kept deliberately
# small: anything domain-ish ("code", "file", "camera") must stay searchable.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
i me my mine myself you your yours we us our ours they them their
is are was were be been being am do does did doing done have has had having
can could will would shall should may might must
what whats which who whom whose when where why how
of in on at to for from with without about into over under again
as by so no not just only very much many more most some any all both each
it its im ive id ill youre dont doesnt didnt cant wont
show tell give find get list see look know remember recall
happen happens happened happening going went
""".split())

# Time words are answered by the retrieval SCOPE (a date window), never by
# matching the literal word inside a summary — no event says "yesterday".
# Dropping them also lets "what did I do today" fall through to the
# chronological path instead of keyword-matching one useless term.
TEMPORAL = frozenset("""
today todays yesterday tomorrow tonight now currently earlier later
recent recently lately ago past previous latest last next morning afternoon evening night
day days week weeks month months year years hour hours minute minutes
""".split())

STOPWORDS = STOPWORDS | TEMPORAL

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-/]*")


def tokenize(query, keep_stopwords=False):
    """Lowercase content terms from `query`, de-duplicated, order preserved."""
    tokens, seen = [], set()
    for match in _TOKEN_RE.findall((query or "").lower()):
        term = match.strip("._-/")
        if len(term) < 2:
            continue
        if not keep_stopwords and term in STOPWORDS:
            continue
        if term in seen:
            continue
        seen.add(term)
        tokens.append(term)
    return tokens


def is_scope_only(query):
    """True when the query carries no content terms (e.g. "what did I do today").

    Such a question is a request for a time/room window, not a keyword lookup.
    """
    return not tokenize(query)

"""The proactive path: decide whether to say something unprompted.

The original version judged from the current minute alone — "is this worth
interrupting for?" with no other context. That question is close to unanswerable,
so it produced noise. This version gives the model the three things that actually
make the judgement possible:

  1. **A stated goal.** If the user opened a focus session, we know what they
     meant to be doing, so drift is detectable instead of guessed at.
  2. **Memory.** Retrieval over what the user did BEFORE the recent past turns
     "you seem stuck" into "you hit this same error on Tuesday and fixed it by
     ...". That is the only kind of unprompted remark that reliably earns its
     interruption.
  3. **Past reactions.** Nudges the user dismissed are shown back to the model so
     it stops re-raising themes that were already rejected.

Restraint is enforced outside the model too: a cooldown, and near-duplicate
suppression over recent nudges (not just exact repeats).

Deliberately still one class and one decision path — with a single revision call
only when a worthwhile draft repeats itself. The parked event-bus/autogen agents
live in agents/_parked/.
"""
import logging
import time
from collections import deque
from threading import Lock

from memory.retrieval.terms import tokenize

logger = logging.getLogger("home_assistant")

# Retrieval must look further back than this, or it just echoes the screen we
# are already looking at instead of recalling prior experience.
RECENT_EXCLUSION_SECONDS = 900
MAX_DIRECT_MEMORIES = 4
MAX_LINKED_MEMORIES = 4
MAX_LINK_ENTITIES = 3

_SYSTEM_PROMPT = """You are the initiative layer of a personal assistant. You receive \
live observations plus relevant memory and decide for yourself whether to speak without \
waiting for a request.

Use your full judgment and creativity. Surface whatever you believe would be valuable in \
this moment: an observation, connection, question, reminder, warning, idea, challenge, \
encouragement, next step, or something the user may not have considered. Those are examples, \
not a list of allowed reasons. Do not force the situation through a fixed rubric of what is \
right, wrong, useful, or permissible; reason from the whole context and the user's likely \
needs. Prefer taking initiative when you have a meaningful contribution.

Look for a non-obvious but grounded connection between the live work and linked memory. When \
memory supplies a useful prior attempt, decision, unresolved thread, preference, or pattern, \
use that connection to make the response more specific and useful. Do not merely recap the \
memory, announce that you remember it, or force a weak connection.

Adapt your tone to the moment. A warning should be calm and direct; concentrated work deserves \
a low-interruption colleague-like tone; visible frustration calls for empathy plus a practical \
move; progress can be met with restrained energy. Infer the best tone from the evidence instead \
of defaulting to cheerful, formal, or motivational language.

Make each contribution feel freshly composed. Start with the substance, not a canned greeting \
or assistant preamble. Vary the opening words, sentence shape, and conversational move from \
recent initiatives. Avoid repeatedly beginning with phrases such as "It looks like", "I \
noticed", "You seem to be", "Hey", or "Just a thought". Never invent novelty by changing facts.

If speaking would add no value right now, reply with exactly: NO ACTION
Otherwise reply naturally for speech. Be direct and reasonably brief (usually 1-3 sentences, \
no formatting)."""


def _similar(a, b, threshold=0.6):
    """Token-overlap similarity — catches rephrased repeats, not just identical ones."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return a.strip().lower() == b.strip().lower()
    return len(ta & tb) / min(len(ta), len(tb)) >= threshold


def _opening(text, words=4):
    """A compact, comparable representation of how a message begins."""
    return " ".join(tokenize(text, keep_stopwords=True)[:words])


def _repeats_opening(text, prior_texts):
    """Catch reused lead-ins while allowing ordinary vocabulary later on."""
    current = tokenize(text, keep_stopwords=True)
    if not current:
        return False
    for prior in prior_texts:
        previous = tokenize(prior, keep_stopwords=True)
        if not previous:
            continue
        # Two repeated lead words catches canned constructions ("it looks",
        # "you are") without rejecting every sentence that happens to start
        # with a common article.
        width = 2 if min(len(current), len(previous)) >= 2 else 1
        if current[:width] == previous[:width]:
            return True
    return False


class ProactiveNarrator:
    def __init__(self, vlm_model, client, cooldown_seconds=300,
                 retriever=None, store_getter=None, focus_cooldown_seconds=120,
                 personal_memory=None):
        self.vlm_model = vlm_model
        self.client = client
        self.cooldown_seconds = cooldown_seconds
        # Drift from a stated goal is worth catching sooner than a general remark.
        self.focus_cooldown_seconds = focus_cooldown_seconds
        self.retriever = retriever
        self.personal_memory = personal_memory
        # Lazy: the graph is connected after the narrator is constructed.
        self._store_getter = store_getter or (lambda: None)
        self._last_spoken_at = 0.0
        self._recent_texts = deque(maxlen=12)
        # Screen, mobile and camera workers run on different threads/event loops.
        # Only one initiative decision should be in flight at a time.
        self._decision_lock = Lock()

    @property
    def store(self):
        try:
            return self._store_getter()
        except Exception:
            return None

    async def consider(self, description, source="screen", context=None):
        """Return {text, kind, focus_id, evidence} for a nudge, or None to stay silent."""
        if not description:
            return None

        # Do not queue stale decisions behind a slow VLM call.
        if not self._decision_lock.acquire(blocking=False):
            logger.debug("Proactive: another decision is already in progress.")
            return None
        try:
            focus = self._active_focus()
            now = time.time()
            cooldown = self.focus_cooldown_seconds if focus else self.cooldown_seconds
            if now - self._last_spoken_at < cooldown:
                logger.debug("Proactive: within delivery cooldown.")
                return None

            recent_texts = self._recent_nudges()
            evidence = self._recall(description, now, focus=focus, context=context)
            personal_context = self._personal_context(description)
            prompt = self._build_prompt(
                description, focus, evidence, source=source, context=context,
                personal_context=personal_context, recent_texts=recent_texts)

            response = await self.client.chat.completions.create(
                model=self.vlm_model,
                messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                max_tokens=180,
                temperature=0.8,
                top_p=0.95,
            )
            text = (response.choices[0].message.content or "").strip()

            if not text or text.upper() == "NO ACTION":
                return None

            repeated_content = any(_similar(text, prior) for prior in recent_texts)
            repeated_opening = _repeats_opening(text, recent_texts)
            if repeated_content or repeated_opening:
                text = await self._revise_repetition(
                    prompt, text, recent_texts, repeated_content, repeated_opening)
                if not text or text.upper() == "NO ACTION":
                    return None

            if any(_similar(text, prior) for prior in recent_texts):
                logger.debug("Proactive: near-duplicate blocked at delivery.")
                return None
            if _repeats_opening(text, recent_texts):
                logger.debug("Proactive: repeated opening blocked at delivery.")
                return None

            self._last_spoken_at = now
            self._recent_texts.append(text)
            return {
                "text": text,
                # `kind` is the sort of nudge this is, not where it came from —
                # that is what `source` below carries. Echoing the source here
                # logged nudges as kind 'camera:IPC-A22E-G' and split what should
                # be one bucket into one per capture device.
                "kind": "focus_drift" if focus else "insight",
                "source": str(source or "unknown"),
                "focus_id": (focus or {}).get("focus_id"),
                "evidence": [
                    {"kind": item.get("kind"), "id": item.get("id"),
                     "text": item.get("text"),
                     "relationship": item.get("relationship")}
                    for item in evidence],
            }
        finally:
            self._decision_lock.release()

    async def _revise_repetition(self, prompt, draft, recent_texts,
                                 repeated_content, repeated_opening):
        """Give a worthwhile draft one chance to become fresh instead of dropping it."""
        requirements = []
        if repeated_content:
            requirements.append(
                "Choose a substantively different insight or conversational move.")
        if repeated_opening:
            requirements.append(
                "Begin with different words and a different sentence shape; do not "
                "solve this by adding a greeting.")
        recent = "\n".join(f"- {_opening(text)}" for text in recent_texts[:8])
        revision_prompt = (
            prompt
            + "\n\nA first draft was:\n"
            + draft
            + "\n\nRevise it because it resembles recent initiatives. "
            + " ".join(requirements)
            + "\nPreserve grounded facts and the tone appropriate to the live moment. "
            + "If no genuinely different contribution is available, reply NO ACTION."
        )
        if recent:
            revision_prompt += "\nRecent openings to avoid:\n" + recent
        response = await self.client.chat.completions.create(
            model=self.vlm_model,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user", "content": revision_prompt}],
            max_tokens=180,
            temperature=0.9,
            top_p=0.95,
        )
        return (response.choices[0].message.content or "").strip()

    # -- context ------------------------------------------------------------
    def _active_focus(self):
        store = self.store
        if store is None:
            return None
        try:
            return store.active_focus_session()
        except Exception as exc:
            logger.debug("Proactive: could not read focus session: %s", exc)
            return None

    def _recall(self, description, now, focus=None, context=None):
        """Relevant old activity plus memories connected through graph entities."""
        if self.retriever is None:
            return []
        cutoff = now - RECENT_EXCLUSION_SECONDS
        query_parts = [str(description)]
        if focus and focus.get("goal"):
            query_parts.append(f"Active goal: {focus['goal']}")
        # Capture sources can supply these when they know more than the visual
        # summary alone. Plumbing identifiers and clip ids are deliberately
        # excluded because they add retrieval noise rather than meaning.
        for key in ("application", "project", "window_title", "camera_name"):
            value = (context or {}).get(key)
            if value:
                query_parts.append(f"{key}: {str(value)[:300]}")
        query = "\n".join(query_parts)
        try:
            direct = self.retriever.retrieve(
                query, limit=6,
                kinds=["event", "entity", "claim", "note", "room"],
                end=cutoff)
        except Exception as exc:
            logger.debug("Proactive: recall failed: %s", exc)
            return []
        return self._expand_linked_memory(direct, cutoff)

    def _expand_linked_memory(self, direct, cutoff):
        """Follow event -> entity -> older event/claim links for useful continuity.

        Text similarity finds a seed. The graph expansion answers the more useful
        question: "what else do we know about the same project, file, person, or
        concept?" Failures are intentionally local so initiative never depends on
        Neo4j being available.
        """
        memories = [dict(item) for item in (direct or [])[:MAX_DIRECT_MEMORIES]]
        store = self.store
        if store is None or not memories:
            return memories

        event_ids = [
            item.get("id") for item in memories
            if item.get("kind") == "event" and item.get("id")
        ]
        entity_refs = []
        for item in memories:
            if item.get("kind") == "entity":
                entity_refs.append({
                    "id": item.get("id"),
                    "name": item.get("title") or item.get("text"),
                })

        if event_ids:
            try:
                by_event = store.entities_for_events(event_ids)
                for item in memories:
                    entities = by_event.get(item.get("id"), [])
                    if entities:
                        item["entities"] = entities
                        entity_refs.extend(
                            {"id": None, "name": entity.get("name")}
                            for entity in entities)
            except Exception as exc:
                logger.debug("Proactive: event link expansion failed: %s", exc)

        unique_entities = []
        seen_entities = set()
        for entity in entity_refs:
            name = str(entity.get("name") or "").strip()
            key = name.casefold()
            if not key or key in seen_entities:
                continue
            seen_entities.add(key)
            unique_entities.append({**entity, "name": name})

        seen_items = {
            (item.get("kind"), item.get("id"), str(item.get("text") or "").casefold())
            for item in memories
        }
        linked = []
        for entity in unique_entities[:MAX_LINK_ENTITIES]:
            try:
                detail = store.entity_detail(entity.get("id") or entity["name"])
            except Exception as exc:
                logger.debug("Proactive: linked memory lookup failed: %s", exc)
                continue
            if not detail:
                continue

            relation = f"linked through {entity['name']}"
            candidates = []
            for claim in detail.get("claims") or []:
                if claim.get("last_seen") is not None and claim["last_seen"] >= cutoff:
                    continue
                candidates.append({
                    "kind": "claim",
                    "id": claim.get("claim_id"),
                    "title": entity["name"],
                    "text": claim.get("text"),
                    "ts": claim.get("last_seen"),
                    "relationship": relation,
                })
            for event in detail.get("events") or []:
                if event.get("span_start") is not None and event["span_start"] >= cutoff:
                    continue
                candidates.append({
                    "kind": "event",
                    "id": event.get("event_id"),
                    "title": event.get("application") or entity["name"],
                    "text": event.get("summary"),
                    "ts": event.get("span_start"),
                    "relationship": relation,
                })
            candidates.sort(key=lambda item: item.get("ts") or 0, reverse=True)
            for item in candidates:
                if not item.get("text"):
                    continue
                key = (
                    item.get("kind"),
                    item.get("id"),
                    str(item.get("text") or "").casefold(),
                )
                if key in seen_items:
                    continue
                seen_items.add(key)
                linked.append(item)
                break
            if len(linked) >= MAX_LINKED_MEMORIES:
                break
        return memories + linked

    def _personal_context(self, description):
        if self.personal_memory is None:
            return ""
        try:
            return self.personal_memory.context(fact_limit=8, query=description)
        except Exception as exc:
            logger.debug("Proactive: personal context failed: %s", exc)
            return ""

    def _recent_nudges(self, limit=10):
        """Recent text survives restarts when the graph store is available."""
        texts = list(reversed(self._recent_texts))
        store = self.store
        if store is not None:
            try:
                texts.extend(
                    row.get("text") for row in store.list_nudges(limit=limit)
                    if row.get("text"))
            except Exception as exc:
                logger.debug("Proactive: could not read recent nudges: %s", exc)
        unique = []
        seen = set()
        for text in texts:
            key = str(text).strip().casefold()
            if key and key not in seen:
                seen.add(key)
                unique.append(str(text).strip())
        return unique[:limit]

    def _dismissed(self):
        store = self.store
        if store is None:
            return []
        try:
            return store.recent_nudge_feedback(limit=8)
        except Exception as exc:
            logger.debug("Proactive: could not read nudge feedback: %s", exc)
            return []

    def _build_prompt(self, description, focus, evidence, source="screen", context=None,
                      personal_context="", recent_texts=None):
        parts = [f"Live observation source: {source}\nObservation:\n{description}"]
        if context:
            lines = "\n".join(f"- {key}: {value}" for key, value in context.items()
                              if value is not None and value != "")
            if lines:
                parts.append("\nSource context:\n" + lines)

        if focus:
            elapsed = (time.time() - (focus.get("started_at") or time.time())) / 60
            parts.append(
                f"\nThe user is in a focus session they started {elapsed:.0f} minutes ago "
                f"(planned {focus.get('planned_minutes')} min).\n"
                f"Their stated goal: {focus.get('goal')}\n"
                "Treat this goal as useful context, not as a rule. Decide what, if "
                "anything, is worth saying.")

        if evidence:
            lines = []
            for item in evidence:
                details = [str(item.get("kind") or "memory")]
                if item.get("relationship"):
                    details.append(item["relationship"])
                entity_names = [
                    entity.get("name") for entity in (item.get("entities") or [])
                    if entity.get("name")
                ]
                if entity_names:
                    details.append("entities: " + ", ".join(entity_names[:5]))
                title = str(item.get("title") or "").strip()
                text = str(item.get("text") or "").strip()
                body = (
                    f"{title}: {text}"
                    if title and title.casefold() not in text.casefold()
                    else text
                )
                lines.append(f"- ({'; '.join(details)}) {body}")
            parts.append(
                "\nRelevant prior and graph-linked memory "
                "(older than the last 15 minutes):\n" + "\n".join(lines))

        if personal_context:
            parts.append(
                "\nRelevant personal preferences and tendencies. Apply only when they "
                "actually help with this moment:\n" + personal_context)

        dismissed = self._dismissed()
        if dismissed:
            lines = "\n".join(
                f"- \"{row.get('text')}\" -> {row.get('feedback')}"
                for row in dismissed)
            parts.append(
                "\nHow the user reacted to recent initiatives. Use this as preference "
                "context while making your own judgment:\n" + lines)

        openings = []
        for text in recent_texts or []:
            opening = _opening(text)
            if opening and opening not in openings:
                openings.append(opening)
        if openings:
            parts.append(
                "\nRecent initiative openings (do not repeat or closely imitate these):\n"
                + "\n".join(f"- {opening}" for opening in openings[:8]))

        parts.append(
            "\nSilently choose the most useful angle and a tone that fits the live "
            "moment. If you speak, lead with specific substance and use linked memory "
            "to advance the thought rather than narrating what is visible.")
        return "\n".join(parts)

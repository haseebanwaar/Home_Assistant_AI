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

Deliberately still one class and one VLM call — the parked event-bus/autogen
agents live in agents/_parked/.
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

_SYSTEM_PROMPT = """You are the initiative layer of a personal assistant. You receive \
live observations plus relevant memory and decide for yourself whether to speak without \
waiting for a request.

Use your full judgment and creativity. Surface whatever you believe would be valuable in \
this moment: an observation, connection, question, reminder, warning, idea, challenge, \
encouragement, next step, or something the user may not have considered. Those are examples, \
not a list of allowed reasons. Do not force the situation through a fixed rubric of what is \
right, wrong, useful, or permissible; reason from the whole context and the user's likely \
needs. Prefer taking initiative when you have a meaningful contribution.

If speaking would add no value right now, reply with exactly: NO ACTION
Otherwise reply naturally for speech. Be direct and reasonably brief (usually 1-3 sentences, \
no formatting)."""


def _similar(a, b, threshold=0.6):
    """Token-overlap similarity — catches rephrased repeats, not just identical ones."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return a.strip().lower() == b.strip().lower()
    return len(ta & tb) / min(len(ta), len(tb)) >= threshold


class ProactiveNarrator:
    def __init__(self, vlm_model, client, cooldown_seconds=300,
                 retriever=None, store_getter=None, focus_cooldown_seconds=120):
        self.vlm_model = vlm_model
        self.client = client
        self.cooldown_seconds = cooldown_seconds
        # Drift from a stated goal is worth catching sooner than a general remark.
        self.focus_cooldown_seconds = focus_cooldown_seconds
        self.retriever = retriever
        # Lazy: the graph is connected after the narrator is constructed.
        self._store_getter = store_getter or (lambda: None)
        self._last_spoken_at = 0.0
        self._recent_texts = deque(maxlen=8)
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

            evidence = self._recall(description, now)
            prompt = self._build_prompt(
                description, focus, evidence, source=source, context=context)

            response = await self.client.chat.completions.create(
                model=self.vlm_model,
                messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                max_tokens=180,
            )
            text = (response.choices[0].message.content or "").strip()

            if not text or text.upper() == "NO ACTION":
                return None
            if any(_similar(text, prior) for prior in self._recent_texts):
                logger.debug("Proactive: near-duplicate blocked at delivery.")
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
                     "text": item.get("text")}
                    for item in evidence],
            }
        finally:
            self._decision_lock.release()

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

    def _recall(self, description, now):
        """Prior activity relevant to what's on screen, excluding the recent past."""
        if self.retriever is None:
            return []
        try:
            return self.retriever.retrieve(
                description, limit=4, kinds=["event", "claim", "note"],
                end=now - RECENT_EXCLUSION_SECONDS)
        except Exception as exc:
            logger.debug("Proactive: recall failed: %s", exc)
            return []

    def _dismissed(self):
        store = self.store
        if store is None:
            return []
        try:
            return store.recent_nudge_feedback(limit=8)
        except Exception as exc:
            logger.debug("Proactive: could not read nudge feedback: %s", exc)
            return []

    def _build_prompt(self, description, focus, evidence, source="screen", context=None):
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
            lines = "\n".join(
                f"- ({item.get('kind')}) {item.get('text')}" for item in evidence)
            parts.append(
                "\nFrom your memory of what this user did previously "
                "(older than the last 15 minutes):\n" + lines)

        dismissed = self._dismissed()
        if dismissed:
            lines = "\n".join(
                f"- \"{row.get('text')}\" -> {row.get('feedback')}"
                for row in dismissed)
            parts.append(
                "\nHow the user reacted to recent initiatives. Use this as preference "
                "context while making your own judgment:\n" + lines)

        return "\n".join(parts)

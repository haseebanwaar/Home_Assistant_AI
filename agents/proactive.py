"""Minimal working proactive path.

Every minute the screen capture produces a description of what's on screen.
This narrator asks the VLM whether that warrants an unprompted spoken nudge,
and returns a short message (or None). A cooldown prevents chatter.

Deliberately small: one class, one VLM call, no event bus or multi-agent
framework. The old event-bus/autogen agents live in agents/_parked/.
"""
import logging
import time

logger = logging.getLogger("home_assistant")

_SYSTEM_PROMPT = """You decide whether to proactively speak to the user based on what is currently on their screen.
Only speak up when it is genuinely useful (a helpful reminder, a spotted problem, a timely suggestion).
Most of the time you should stay silent.
If nothing is worth saying, reply with exactly: NO ACTION
Otherwise reply with a single short, friendly spoken sentence (max 2 sentences, no formatting)."""


class ProactiveNarrator:
    def __init__(self, vlm_model, client, cooldown_seconds=300):
        self.vlm_model = vlm_model
        self.client = client
        self.cooldown_seconds = cooldown_seconds
        self._last_spoken_at = 0.0
        self._last_text = ""

    async def consider(self, description):
        """Return a short spoken message, or None to stay silent."""
        if not description:
            return None

        now = time.time()
        if now - self._last_spoken_at < self.cooldown_seconds:
            logger.debug("Proactive: within cooldown, staying silent.")
            return None

        response = await self.client.chat.completions.create(
            model=self.vlm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Current screen activity:\n{description}"},
            ],
            max_tokens=80,
        )
        text = (response.choices[0].message.content or "").strip()

        if not text or "NO ACTION" in text.upper():
            return None
        if text == self._last_text:
            return None

        self._last_spoken_at = now
        self._last_text = text
        return text

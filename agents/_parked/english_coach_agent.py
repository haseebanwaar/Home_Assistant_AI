# agents/english_coach_agent.py
import asyncio
from agents.base_agent import AsyncAgent


class EnglishCoachAgent(AsyncAgent):
    def __init__(self, vlm, event_bus):
        super().__init__("EnglishCoachAgent", vlm, event_bus)
        event_bus.subscribe("user_message", self.handle_message)

    async def handle_message(self, data):
        text = data.get("text", "")
        if not text.strip():
            return

        # Example: Ask VLM/LLM for grammar analysis and improvement
        prompt = f"""
        You are an English language coach.
        The user said: "{text}"
        1. Identify any grammatical or phrasing mistakes.
        2. Suggest a more natural version.
        3. Give a one-line encouragement or micro-tip.
        Respond concisely in under 3 sentences.
        """
        feedback = await self.vlm.ainvoke(prompt)  # assume async call
        await self.event_bus.publish("coach_feedback", {"text": feedback})
        # Optionally: also make it a proactive message
        await self.event_bus.publish("proactive_insight", {"text": feedback})

    async def run(self):
        # No continuous loop needed — reacts to messages only
        while True:
            await asyncio.sleep(3600)
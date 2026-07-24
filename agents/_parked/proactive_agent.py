import asyncio
from datetime import datetime, timedelta
from agents.base_agent import AsyncAgent

class ProactiveAgent(AsyncAgent):
    def __init__(self, vlm, event_bus):
        super().__init__("ProactiveAgent", vlm, event_bus)
        self.last_insight_time = datetime.min
        self.last_message = ""
        self.cooldown = timedelta(minutes=5)

        # Subscribe to relevant system events
        event_bus.subscribe("event_extracted", self.handle_event)
        event_bus.subscribe("memory_summary", self.handle_summary)
        event_bus.subscribe("face_recognized", self.handle_face)

    async def handle_event(self, data):
        description = data.get("description", "")
        await self.consider_proactive_message(f"Event detected: {description}")

    async def handle_summary(self, data):
        summary = data.get("summary", "")
        await self.consider_proactive_message(f"Recent activity summary: {summary}")

    async def handle_face(self, data):
        names = data.get("names", [])
        if names:
            await self.consider_proactive_message(f"You’re with {', '.join(names)} right now.")

    async def consider_proactive_message(self, context_text):
        # Enforce cooldown
        if datetime.now() - self.last_insight_time < self.cooldown:
            return

        prompt = f"""
        You are the proactive assistant core.
        Context: {context_text}
        Determine if this warrants a spoken insight.
        If yes, craft a short, friendly message (max 2 sentences).
        If not, respond with "NO ACTION".
        """

        response = await self.vlm.ainvoke(prompt)
        if "NO ACTION" not in response and response.strip() != self.last_message:
            self.last_insight_time = datetime.now()
            self.last_message = response.strip()
            await self.event_bus.publish("proactive_insight", {"text": response.strip()})

    async def run(self):
        # Occasionally generate summaries or reminders
        while True:
            await asyncio.sleep(600)  # every 10 minutes
            await self.consider_proactive_message("Idle system check.")

from agents.base_agent import AsyncAgent


class EventExtractorAgent(AsyncAgent):
    def __init__(self, vlm, event_bus):
        super().__init__("EventExtractorAgent", vlm, event_bus)
        event_bus.subscribe("new_frame", self.handle_frame)

    async def handle_frame(self, data):
        # analyze embeddings for scene change, new person, etc.
        # if detected, send a proactive insight
        await self.event_bus.publish("proactive_insight", {
            "text": "User started watching YouTube."
        })
from agents.base_agent import AsyncAgent


class TalkerAgent(AsyncAgent):
    def __init__(self, vlm, event_bus, tts_func):
        super().__init__("TalkerAgent", vlm,vlm, event_bus)
        self.tts_func = tts_func
        event_bus.subscribe("proactive_insight", self.respond)

    async def respond(self, data):

        reply = data["text"]
        await self.tts_func(reply)  # speak it
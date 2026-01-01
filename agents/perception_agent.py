from agents.base_agent import AsyncAgent


class PerceptionAgent(AsyncAgent):
    def __init__(self, vlm, event_bus, capture_func, embed_func, qdrant_client):
        super().__init__("PerceptionAgent", vlm,vlm, event_bus)
        self.capture_func = capture_func
        self.embed_func = embed_func
        self.qdrant = qdrant_client

    async def respond(self, data):
        reply = data["text"]
        await self.tts_func(reply)  # speak it
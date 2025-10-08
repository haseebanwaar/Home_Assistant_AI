# agents/vision_memory_agent.py
import asyncio
from datetime import datetime
import time
from agents.base_agent import AsyncAgent


class PerceptionAgent(AsyncAgent):
    def __init__(self, vlm, event_bus, capture_func, embed_func, qdrant_client):
        super().__init__("PerceptionAgent", vlm, event_bus)
        self.capture_func = capture_func
        self.embed_func = embed_func
        self.qdrant = qdrant_client

    async def run(self):
        while True:
            frame = await self.capture_func()
            embedding = await self.embed_func(frame)
            self.qdrant.log_activity("vision_memory", embedding, metadata={"time": datetime.now().isoformat()})
            await self.event_bus.publish("new_frame", {"timestamp": datetime.now(), "embedding": embedding})
            await asyncio.sleep(3)  # every few seconds
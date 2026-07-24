# agents/base_async_agent.py
import asyncio

class AsyncAgent:
    def __init__(self, name, client,vlm, event_bus):
        self.name = name
        self.client = client
        self.vlm = vlm
        self.event_bus = event_bus

    async def run(self):
        """Override this with your agent loop"""
        raise NotImplementedError
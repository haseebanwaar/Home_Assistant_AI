import asyncio

class EventBus:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.subscribers = {}

    def subscribe(self, event_type, callback):
        self.subscribers.setdefault(event_type, []).append(callback)

    async def publish(self, event_type, data):
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                asyncio.create_task(callback(data))

    async def run_forever(self):
        while True:
            event_type, data = await self.queue.get()
            await self.publish(event_type, data)
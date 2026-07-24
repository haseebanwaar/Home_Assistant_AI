"""A tiny tool registry for the POC.

This is deliberately simple: one place to declare callable tools, exposed to the
VLM through the standard OpenAI function-calling schema. To add a new capability
(calendar, web search, Tuya smart devices, ...) register another Tool here — no
MCP server or extra protocol layer needed for a single local user.
"""
import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger("home_assistant")


@dataclass
class Tool:
    name: str
    schema: dict           # the OpenAI "function" object (name/description/parameters)
    handler: Callable      # sync or async fn(**args) -> str | list[str]

    async def run(self, **kwargs):
        """Run the handler, off-thread if it's blocking."""
        if inspect.iscoroutinefunction(self.handler):
            return await self.handler(**kwargs)
        return await asyncio.to_thread(self.handler, **kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, schema, handler):
        self._tools[name] = Tool(name=name, schema=schema, handler=handler)
        logger.info("Registered tool: %s", name)

    def get(self, name):
        return self._tools.get(name)

    @property
    def openai_schemas(self):
        """The tools payload for client.chat.completions.create(tools=...)."""
        return [{"type": "function", "function": t.schema} for t in self._tools.values()]

    @property
    def names(self):
        return list(self._tools)

    def __bool__(self):
        return bool(self._tools)


# --- Built-in tool schemas ---------------------------------------------------

RETRIEVE_MEMORY_SCHEMA = {
    "name": "retrieve_memory",
    "description": "Search the user's own past screen/activity history from the vector store.",
    "parameters": {
        "type": "object",
        "properties": {
            "search_query": {
                "type": "string",
                "description": "The topic to look for. String value for semantic search in vector db.",
            },
            "time_value": {
                "type": "number",
                "description": "The numerical value for the time range (e.g., 2.5, 10, 1).",
            },
            "time_unit": {
                "type": "string",
                "enum": ["minutes", "hours", "days", "weeks", "months"],
                "description": "The unit of time to look back.",
            },
        },
    },
}


def register_default_tools(registry: ToolRegistry, past_memory):
    """Register the tools available to the POC assistant.

    Add more tools here as the POC grows, e.g.:
        registry.register("control_light", LIGHT_SCHEMA, tuya_set_light)
        registry.register("web_search", SEARCH_SCHEMA, run_web_search)
    """
    registry.register("retrieve_memory", RETRIEVE_MEMORY_SCHEMA, past_memory.retrieve_memory)

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

from providers.local_openAI import model_llm


async def main() -> None:
    model_client = model_llm
    agent = AssistantAgent(
        "assistant",
        tools=[tool],
        model_client=model_client,
        system_message="You are an AI assistant.",
    )
    await Console(
        agent.on_messages_stream(
            # [TextMessage(content="1- Search the web for Allama Iqbal.\n2- Create a new file locally with name iqbal.txt.\n3- write the resulting text of search in iqbal.txt", source="user")], CancellationToken()
            [TextMessage(content=content, source="user")], CancellationToken()
        )
    )


asyncio.run(main())

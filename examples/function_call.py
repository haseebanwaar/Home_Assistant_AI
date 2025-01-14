import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from providers.local_openAI import model_llm, model_vlm


async def add_two_numbers(a: int, b: int) -> int:
    return a + b


async def multiply_two_numbers(a: int, b: int) -> int:
    return a * b


async def main() -> None:
    # Define an agent
    weather_agent = AssistantAgent(
        name="weather_agent",
        model_client=model_llm,
        tools=[multiply_two_numbers, add_two_numbers],
    )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")
    # termination = TextMentionTermination("STOP")

    # Define a team
    agent_team = RoundRobinGroupChat([weather_agent], termination_condition=termination, max_turns=12)

    # Run the team and stream messages to the console
    stream = agent_team.run_stream(task="What 5 plus 6?")
    await Console(stream)


asyncio.run(main())

import asyncio
import time
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
# lmdeploy serve api_server F:\try\qwen\InternVL2_5-8B-AWQ --server-port 23333 --session-len 65032 --cache-max-entry-count 0.85 --quant-policy 4
from providers.local_openAI import model_llm,model_vlm

tim = time.perf_counter()
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_llm,
    # model_client=model_vlm,
)
tokens = 0


async def assistant_run() -> None:
    global tokens
    response = await agent.on_messages(
        [TextMessage(content="what the difference and similarities between tiger, lion. provide a 500 word answer",
                     source="user")],
        cancellation_token=CancellationToken(),
    )
    # print(response.inner_messages)
    tokens = response.chat_message.models_usage.completion_tokens


asyncio.run(assistant_run())
print(tokens / (time.perf_counter() - tim))

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
# transformer 4.57,   torch 2.9.0 ,  triton 3.5.0,
# source /home/haseeb/venv2/bin/activate

# model_name_vlm = client.models.list().data[0].id

load_dotenv()

def new_vlm_client() -> AsyncOpenAI:
    """A fresh AsyncOpenAI pointed at the VLM server.

    Each background capture worker (screen, and one per camera) drives the VLM
    from its own thread via asyncio.run. httpx's async connection pool is bound to
    the loop that first used it, so sharing ONE client across concurrent worker
    threads can corrupt that pool. Give each concurrent worker its own client so
    every client stays single-threaded.
    """
    return AsyncOpenAI(
        api_key=os.getenv("VLM_API_KEY"),
        # Qwen3.6 is served by llama.cpp; configure its OpenAI endpoint in .env.
        base_url=os.getenv("VLM_BASE_URL", "http://127.0.0.1:8888/v1"),
    )


def thinking_request_kwargs(enabled=False, budget=None):
    """llama.cpp/Qwen chat-template controls for one request."""
    kwargs = {"extra_body": {
        "chat_template_kwargs": {"enable_thinking": bool(enabled)},
    }}
    if enabled and budget is not None and int(budget) >= 0:
        kwargs["extra_body"]["thinking_budget_tokens"] = int(budget)
    return kwargs


# Shared client for the main app loop and the (single) screen worker thread.
client = new_vlm_client()

# async getter instead of a global variable
async def get_model_name_vlm() -> str:
    models = await client.models.list()
    return models.data[0].id

# model_name = await get_model_name_vlm()



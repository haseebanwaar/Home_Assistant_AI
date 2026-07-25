import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
 # vllm serve "/mnt/d/models/vlm/qwen3vl_8b" --max-model-len 20000 --kv-cache-memory-bytes 6G --trust-remote-code --max_num_seqs 1 --enable-auto-tool-choice --tool-call-parser hermes --no-enable-prefix-caching --limit-mm-per-prompt.video=1 --enforce-eager --video_pruning_rate 0.1 --dtype half  --limit-mm-per-prompt 1  --mm-processor-cache-gb 1 --async-scheduling
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
        # Alternates: lmdeploy :23333, llama.cpp :8080, vllm :8000 — set VLM_BASE_URL in .env
        base_url=os.getenv("VLM_BASE_URL"),
    )


# Shared client for the main app loop and the (single) screen worker thread.
client = new_vlm_client()

# async getter instead of a global variable
async def get_model_name_vlm() -> str:
    models = await client.models.list()
    return models.data[0].id

# model_name = await get_model_name_vlm()



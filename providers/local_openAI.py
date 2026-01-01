import asyncio

from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI
from openai import OpenAI
import json
 # vllm serve "/mnt/d/models/vlm/qwen3vl_8b" --max-model-len 20000 --kv-cache-memory-bytes 6G --trust-remote-code --max_num_seqs 1 --enable-auto-tool-choice --tool-call-parser hermes --no-enable-prefix-caching --limit-mm-per-prompt.video=1 --enforce-eager --video_pruning_rate 0.1 --dtype half  --limit-mm-per-prompt 1  --mm-processor-cache-gb 1 --async-scheduling
#
# client = OpenAI(api_key='ss', base_url='http://localhost:8000/v1')
# source /home/haseeb/venv2/bin/activate

# model_name_vlm = client.models.list().data[0].id

client = AsyncOpenAI(
    api_key="ss",
    # base_url="http://localhost:23333/v1" #this is lmdeploy
    # base_url="http://localhost:8080/v1" #this is llamacpp
    base_url="http://localhost:8000/v1" #this is vllm
)

# async getter instead of a global variable
async def get_model_name_vlm() -> str:
    models = await client.models.list()
    return models.data[0].id

# model_name = await get_model_name_vlm()









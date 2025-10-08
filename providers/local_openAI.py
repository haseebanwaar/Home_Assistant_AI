import asyncio

from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI
# from openai import OpenAI
import openai
import json

# client = OpenAI(api_key='ss', base_url='http://localhost:23333/v1')
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










--rope-scaling '{"rope_type":"yarn","factor":3.0,"original_max_position_embeddings": 262144,"mrope_section":[24,20,20],"mrope_interleaved": true}' --max-model-len 1000000








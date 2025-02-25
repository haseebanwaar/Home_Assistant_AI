from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import OpenAI

model_vlm_local = OpenAI(api_key='ss', base_url='http://localhost:23333/v1')
model_name = model_vlm_local.models.list().data[0].id

model_llm = OpenAIChatCompletionClient(
    model="F:\\try\\qwen\\Qwen2.5-1.5B-Instruct-AWQ",
    base_url="http://localhost:23333/v1", api_key="placeholder",
    model_capabilities={"vision": True, "function_calling": True, "json_output": True})
model_vlm = OpenAIChatCompletionClient(
    model="F:\\try\\qwen\\InternVL2_5-8B-AWQ",
    base_url="http://localhost:23334/v1", api_key="placeholder",
    model_capabilities={"vision": True, "function_calling": True, "json_output": True})

model_function_callling = OpenAIChatCompletionClient(
    model="F:\\try\\qwen\\Qwen2.5-1.5B-Instruct-AWQ",
    base_url="http://localhost:23333/v1", api_key="placeholder",
    model_capabilities={"vision": False, "function_calling": True, "json_output": True})



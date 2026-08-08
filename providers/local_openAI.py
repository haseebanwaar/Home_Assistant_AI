import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from utils.jobs import VLM, describe_frames, jobs
# transformer 4.57,   torch 2.9.0 ,  triton 3.5.0,
# source /home/haseeb/venv2/bin/activate

# model_name_vlm = client.models.list().data[0].id

load_dotenv()

class _TrackedStream:
    """A streaming completion that keeps its job open until the tokens stop.

    The GPU is busy for the whole stream, not just for the call that opens it, so
    finishing the job on `create()` would report a two-minute spoken answer as a
    50 ms request.
    """

    def __init__(self, stream, job_id):
        self._stream = stream
        self._job_id = job_id

    def __getattr__(self, name):
        return getattr(self._stream, name)

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                yield chunk
        except BaseException as exc:  # noqa: BLE001 - recorded then re-raised
            jobs.finish(self._job_id, "error", f"{type(exc).__name__}: {exc}"[:160])
            raise
        else:
            jobs.finish(self._job_id)

    async def close(self):
        jobs.finish(self._job_id, "cancelled")
        await self._stream.close()

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        jobs.finish(self._job_id, "cancelled")
        return await self._stream.__aexit__(*args)


class _TrackedCompletions:
    """`client.chat.completions` with every request on the jobs board.

    Wrapping here rather than at each call site is deliberate: there are a dozen
    callers across the API, the capture threads and the agents, and one of them
    forgetting to register is exactly the request that would go unexplained.

    `job_label` names the work in the UI; it is consumed here and never sent to
    the server.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def create(self, *args, job_label=None, **kwargs):
        job_id = jobs.start(VLM, job_label or "VLM request",
                            describe_frames(kwargs.get("messages")))
        try:
            result = await self._inner.create(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - recorded then re-raised
            jobs.finish(job_id, "error", f"{type(exc).__name__}: {exc}"[:160])
            raise
        if kwargs.get("stream"):
            return _TrackedStream(result, job_id)
        jobs.finish(job_id)
        return result


class _TrackedChat:
    def __init__(self, inner):
        self._inner = inner
        self.completions = _TrackedCompletions(inner.completions)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TrackedVlmClient:
    """An AsyncOpenAI whose chat completions are visible on the jobs board.

    Everything else (`models.list()`, `audio`, ...) passes straight through.
    """

    def __init__(self, inner):
        self._inner = inner
        self.chat = _TrackedChat(inner.chat)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def new_vlm_client() -> TrackedVlmClient:
    """A fresh AsyncOpenAI pointed at the VLM server.

    Each background capture worker (screen, and one per camera) drives the VLM
    from its own thread via asyncio.run. httpx's async connection pool is bound to
    the loop that first used it, so sharing ONE client across concurrent worker
    threads can corrupt that pool. Give each concurrent worker its own client so
    every client stays single-threaded.
    """
    return TrackedVlmClient(AsyncOpenAI(
        api_key=os.getenv("VLM_API_KEY"),
        # Qwen3.6 is served by llama.cpp; configure its OpenAI endpoint in .env.
        base_url=os.getenv("VLM_BASE_URL", "http://127.0.0.1:8888/v1"),
    ))


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



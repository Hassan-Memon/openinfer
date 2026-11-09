"""The official `openai` Python SDK must work unchanged against OpenInfer.

The SDK is pointed at the ASGI app through an httpx transport, so no socket is opened.
"""

from __future__ import annotations

import httpx
import pytest
from openai import AsyncOpenAI, NotFoundError


@pytest.fixture
async def sdk(app):
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    client = AsyncOpenAI(
        base_url="http://testserver/v1", api_key="not-needed", http_client=http_client
    )
    try:
        yield client
    finally:
        await http_client.aclose()


async def test_sdk_models_list(sdk: AsyncOpenAI):
    models = await sdk.models.list()
    ids = [m.id for m in models.data]
    assert "fast" in ids
    assert "echo/echo" in ids


async def test_sdk_chat_completion(sdk: AsyncOpenAI):
    completion = await sdk.chat.completions.create(
        model="fast",
        messages=[{"role": "user", "content": "Hello from the SDK"}],
        temperature=0,
    )
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content == "Echo: Hello from the SDK"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage is not None and completion.usage.total_tokens > 0
    assert completion.model == "echo"


async def test_sdk_chat_streaming(sdk: AsyncOpenAI):
    stream = await sdk.chat.completions.create(
        model="echo",
        messages=[{"role": "user", "content": "stream me please"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    text = ""
    finish = None
    usage = None
    async for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.choices:
            text += chunk.choices[0].delta.content or ""
            finish = chunk.choices[0].finish_reason or finish
    assert text == "Echo: stream me please"
    assert finish == "stop"
    assert usage is not None and usage.completion_tokens > 0


async def test_sdk_tool_call(sdk: AsyncOpenAI):
    completion = await sdk.chat.completions.create(
        model="echo",
        messages=[{"role": "user", "content": "what time is it"}],
        tools=[
            {"type": "function", "function": {"name": "clock", "parameters": {"type": "object"}}}
        ],
        tool_choice={"type": "function", "function": {"name": "clock"}},
    )
    call = completion.choices[0].message.tool_calls[0]
    assert call.function.name == "clock"
    assert call.function.arguments == "{}"


async def test_sdk_embeddings(sdk: AsyncOpenAI):
    result = await sdk.embeddings.create(model="embed", input=["alpha", "beta"])
    assert len(result.data) == 2
    assert result.data[0].index == 0
    assert len(result.data[0].embedding) == 384
    assert result.usage.prompt_tokens > 0


async def test_sdk_unknown_model_raises_not_found(sdk: AsyncOpenAI):
    with pytest.raises(NotFoundError):
        await sdk.chat.completions.create(
            model="does-not-exist", messages=[{"role": "user", "content": "x"}]
        )

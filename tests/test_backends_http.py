"""Adapter tests for the HTTP backends using httpx.MockTransport (no network)."""

from __future__ import annotations

import json

import httpx
import pytest

from openinfer.backends.cloud import PRESETS, CloudBackend
from openinfer.backends.ollama import OllamaBackend
from openinfer.backends.vllm import VLLMBackend
from openinfer.config import BackendConfig
from openinfer.errors import BackendError, BackendTimeout, BackendUnavailable, RateLimited
from openinfer.types import ChatRequest, EmbeddingRequest


def chat_req(model: str = "m") -> ChatRequest:
    return ChatRequest(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=5)


def cloud(handler, preset: str = "groq") -> CloudBackend:
    cfg = BackendConfig(type="cloud", preset=preset, api_key="k", models=["m"])
    return CloudBackend("cloud", cfg, transport=httpx.MockTransport(handler))


# -- OpenAI compatible (cloud / vllm) -----------------------------------------------------------


async def test_cloud_chat_sends_openai_body_and_parses_reply():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": "m-2024",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "yo"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    b = cloud(handler)
    result = await b.chat(chat_req())
    assert seen["url"] == PRESETS["groq"].base_url + "/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "m" and seen["body"]["max_tokens"] == 5
    assert "stream" not in seen["body"]
    assert result.content == "yo"
    assert result.model == "m-2024"
    assert result.usage.prompt_tokens == 3 and result.usage.exact is True
    await b.aclose()


async def test_cloud_stream_parses_sse_and_usage():
    sse = "\n".join(
        [
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}',
            "",
            'data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}',
            "",
            'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}',
            "",
            'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}',
            "",
            "data: [DONE]",
            "",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )

    b = cloud(handler)
    chunks = [c async for c in b.chat_stream(chat_req())]
    assert "".join(c.delta or "" for c in chunks) == "Hello"
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 2
    assert any(c.finish_reason == "stop" for c in chunks)
    await b.aclose()


async def test_gemini_preset_disables_stream_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "stream_options" not in body
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    b = cloud(handler, preset="gemini")
    assert b.base_url.startswith("https://generativelanguage.googleapis.com")
    assert [c async for c in b.chat_stream(chat_req())] == []
    await b.aclose()


@pytest.mark.parametrize(
    ("status", "exc_type", "retriable"),
    [(429, RateLimited, True), (503, BackendError, True), (400, BackendError, False)],
)
async def test_cloud_http_errors_map_to_backend_errors(status, exc_type, retriable):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": f"nope {status}"}})

    b = cloud(handler)
    with pytest.raises(exc_type) as exc:
        await b.chat(chat_req())
    assert exc.value.retriable is retriable
    assert f"nope {status}" in exc.value.message
    await b.aclose()


async def test_cloud_stream_error_status_is_raised_before_any_chunk():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow"}})

    b = cloud(handler)
    with pytest.raises(RateLimited):
        async for _ in b.chat_stream(chat_req()):
            pass
    await b.aclose()


async def test_cloud_connection_and_timeout_errors():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(BackendUnavailable) as exc:
        await cloud(refuse).chat(chat_req())
    assert exc.value.retriable is False
    with pytest.raises(BackendTimeout) as exc2:
        await cloud(slow).chat(chat_req())
    assert exc2.value.retriable is True


async def test_cloud_embeddings_and_health():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            body = json.loads(request.content)
            assert body["input"] == ["a", "b"]
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ],
                    "model": "emb",
                    "usage": {"prompt_tokens": 2},
                },
            )
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    b = cloud(handler)
    result = await b.embed(EmbeddingRequest(model="emb", inputs=["a", "b"]))
    assert result.vectors == [[1.0, 0.0], [0.0, 1.0]]  # sorted by index
    status = await b.health()
    assert status.ok and "1 models" in status.detail
    await b.aclose()


async def test_vllm_backend_is_local_and_free():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    cfg = BackendConfig(type="vllm", base_url="http://vllm:8001/v1", models=["q"])
    b = VLLMBackend("vllm", cfg, transport=httpx.MockTransport(handler))
    assert b.local is True and b.pricing_provider is None
    assert (await b.health()).ok
    await b.aclose()


def test_unknown_cloud_preset_without_base_url_fails_clearly():
    cfg = BackendConfig(type="cloud", preset="mystery", api_key="k")
    with pytest.raises(Exception, match="unknown preset"):
        CloudBackend("c", cfg)


# -- Ollama --------------------------------------------------------------------------------------


def ollama(handler) -> OllamaBackend:
    cfg = BackendConfig(type="ollama", base_url="http://ollama:11434", models=["q"])
    return OllamaBackend("ollama", cfg, transport=httpx.MockTransport(handler))


async def test_ollama_chat_uses_native_api_and_exact_counts():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["options"]["num_predict"] == 5
        return httpx.Response(
            200,
            json={
                "model": "q",
                "message": {"role": "assistant", "content": "hey", "thinking": "hmm"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 7,
                "eval_count": 2,
            },
        )

    b = ollama(handler)
    result = await b.chat(chat_req("q"))
    assert result.content == "hey"
    assert result.reasoning == "hmm"
    assert result.usage.prompt_tokens == 7 and result.usage.completion_tokens == 2
    assert result.usage.exact is True
    await b.aclose()


async def test_ollama_stream_ndjson():
    lines = [
        {"message": {"role": "assistant", "content": "a"}, "done": False},
        {"message": {"role": "assistant", "content": "b"}, "done": False},
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 3,
            "eval_count": 2,
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    b = ollama(handler)
    chunks = [c async for c in b.chat_stream(chat_req("q"))]
    assert "".join(c.delta or "" for c in chunks) == "ab"
    assert chunks[-1].finish_reason == "length"
    assert chunks[-1].usage.completion_tokens == 2
    await b.aclose()


async def test_ollama_tool_calls_are_converted_to_openai_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}],
                },
                "done": True,
            },
        )

    result = await ollama(handler).chat(chat_req("q"))
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0]["function"]["arguments"] == '{"x": 1}'
    assert result.tool_calls[0]["type"] == "function"


async def test_ollama_missing_model_is_not_retriable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'q' not found"})

    with pytest.raises(BackendError) as exc:
        await ollama(handler).chat(chat_req("q"))
    assert exc.value.retriable is False
    assert "ollama pull" in exc.value.message


async def test_ollama_embed_and_health():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(
                200, json={"model": "e", "embeddings": [[0.1, 0.2]], "prompt_eval_count": 2}
            )
        return httpx.Response(200, json={"models": [{"name": "q:latest"}]})

    b = ollama(handler)
    result = await b.embed(EmbeddingRequest(model="e", inputs=["x"]))
    assert result.vectors == [[0.1, 0.2]] and result.usage.exact is True
    status = await b.health()
    assert status.ok and "1 models" in status.detail
    assert await b.list_pulled_models() == ["q:latest"]
    await b.aclose()


async def test_ollama_down_is_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    b = ollama(handler)
    with pytest.raises(BackendUnavailable):
        await b.chat(chat_req("q"))
    assert (await b.health()).ok is False

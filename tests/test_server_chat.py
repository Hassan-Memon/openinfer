"""HTTP level tests for /v1/chat/completions and friends, on the echo backend."""

from __future__ import annotations

import json

import httpx
import pytest

from openinfer.metrics import Metrics
from openinfer.server import create_app
from tests.conftest import make_config


async def collect_sse(response: httpx.Response) -> list[str]:
    events = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            events.append(line[5:].strip())
    return events


async def test_health(client: httpx.AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backends"]["echo"]["ok"] is True
    assert body["backends"]["echo"]["local"] is True


async def test_health_shallow(client: httpx.AsyncClient):
    r = await client.get("/health", params={"deep": "false"})
    assert r.status_code == 200
    assert "backends" not in r.json()


async def test_list_models(client: httpx.AsyncClient):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "fast" in ids and "echo/echo" in ids
    one = await client.get("/v1/models/echo/echo")
    assert one.status_code == 200
    assert one.json()["openinfer"]["kind"] == "chat"
    missing = await client.get("/v1/models/nope")
    assert missing.status_code == 404


async def test_chat_completion_shape(client: httpx.AsyncClient):
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "fast", "messages": [{"role": "user", "content": "hello world"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["model"] == "echo"
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": "Echo: hello world"}
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    assert body["openinfer"]["backend"] == "echo"
    assert body["openinfer"]["alias"] == "fast"
    assert r.headers["x-openinfer-backend"] == "echo"
    assert r.headers["x-openinfer-attempts"] == "1"


async def test_chat_completion_content_parts(client: httpx.AsyncClient):
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "echo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "multi"},
                        {"type": "text", "text": "part"},
                    ],
                }
            ],
        },
    )
    assert r.json()["choices"][0]["message"]["content"] == "Echo: multipart"


async def test_chat_stream_sse(client: httpx.AsyncClient):
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "fast",
            "stream": True,
            "messages": [{"role": "user", "content": "one two three"}],
        },
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = await collect_sse(r)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[0]["openinfer"]["backend"] == "echo"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Echo: one two three"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert len({c["id"] for c in chunks}) == 1


async def test_chat_stream_include_usage(client: httpx.AsyncClient):
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "echo",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as r:
        events = await collect_sse(r)
    usage_chunk = json.loads(events[-2])
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["completion_tokens"] > 0


async def test_forced_tool_call(client: httpx.AsyncClient):
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "echo",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": tools,
            "tool_choice": "required",
        },
    )
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] is None
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"


async def test_unknown_model_is_openai_style_404(client: httpx.AsyncClient):
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-99", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "ModelNotFound"
    assert "gpt-99" in err["message"]


async def test_validation_error_is_400(client: httpx.AsyncClient):
    r = await client.post("/v1/chat/completions", json={"model": "echo"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "validation_error"
    r = await client.post("/v1/chat/completions", json={"model": "echo", "messages": []})
    assert r.status_code == 400


async def test_stream_routing_error_returns_json_not_sse(client: httpx.AsyncClient):
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "nope", "stream": True, "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


PRIVACY_YAML = """
ledger: {url: "sqlite:///:memory:"}
backends:
  echo: {type: echo}
  cloud:
    type: cloud
    preset: groq
    api_key: test-key
    models: [llama-3.1-8b-instant]
aliases:
  fast: [cloud/llama-3.1-8b-instant, echo/echo]
  cloudonly: [cloud/llama-3.1-8b-instant]
policy: {retries: 0}
"""


@pytest.fixture
async def privacy_client():
    app = create_app(make_config(PRIVACY_YAML), metrics=Metrics(enabled=False))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_private_header_skips_cloud(privacy_client: httpx.AsyncClient):
    r = await privacy_client.post(
        "/v1/chat/completions",
        headers={"X-OpenInfer-Tags": "private"},
        json={"model": "fast", "messages": [{"role": "user", "content": "secret"}]},
    )
    assert r.status_code == 200
    assert r.json()["openinfer"]["backend"] == "echo"
    assert r.json()["openinfer"]["attempts"] == 1  # cloud was never tried


async def test_private_via_body_object_fails_closed_on_cloud_only(
    privacy_client: httpx.AsyncClient,
):
    r = await privacy_client.post(
        "/v1/chat/completions",
        json={
            "model": "cloudonly",
            "messages": [{"role": "user", "content": "secret"}],
            "openinfer": {"tags": ["private"]},
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "NoCandidates"


async def test_private_via_metadata(privacy_client: httpx.AsyncClient):
    r = await privacy_client.post(
        "/v1/chat/completions",
        json={
            "model": "fast",
            "messages": [{"role": "user", "content": "secret"}],
            "metadata": {"privacy": "private"},
        },
    )
    assert r.status_code == 200
    assert r.json()["openinfer"]["backend"] == "echo"


async def test_max_cost_header_routes_to_free_local(privacy_client: httpx.AsyncClient):
    r = await privacy_client.post(
        "/v1/chat/completions",
        headers={"X-OpenInfer-Max-Cost-USD": "0"},
        json={"model": "fast", "messages": [{"role": "user", "content": "cheap please"}]},
    )
    assert r.status_code == 200
    assert r.json()["openinfer"]["backend"] == "echo"
    assert r.json()["openinfer"]["attempts"] == 1


AUTH_YAML = """
server: {api_key: s3cret}
ledger: {url: "sqlite:///:memory:"}
backends:
  echo: {type: echo}
aliases:
  fast: [echo/echo]
"""


async def test_bearer_auth():
    app = create_app(make_config(AUTH_YAML), metrics=Metrics(enabled=False))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        body = {"model": "fast", "messages": [{"role": "user", "content": "x"}]}
        assert (await c.post("/v1/chat/completions", json=body)).status_code == 401
        assert (
            await c.post(
                "/v1/chat/completions", json=body, headers={"Authorization": "Bearer wrong"}
            )
        ).status_code == 401
        ok = await c.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer s3cret"}
        )
        assert ok.status_code == 200
        # health stays open for load balancers
        assert (await c.get("/health", params={"deep": "false"})).status_code == 200


async def test_ledger_endpoint(client: httpx.AsyncClient):
    await client.post(
        "/v1/chat/completions",
        json={"model": "fast", "messages": [{"role": "user", "content": "x"}]},
    )
    r = await client.get("/v1/ledger", params={"hours": 1, "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["totals"]["requests"] == 1
    assert body["by_backend"][0]["backend"] == "echo"
    assert body["recent"][0]["endpoint"] == "chat"


async def test_hardware_endpoint(client: httpx.AsyncClient):
    r = await client.get("/v1/hardware")
    assert r.status_code == 200
    body = r.json()
    assert {"os", "cpu", "accelerators", "recommended"} <= set(body)
    assert body["recommended"]["backend"] == "llamacpp"


async def test_root_redirects_to_docs(client: httpx.AsyncClient):
    r = await client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"

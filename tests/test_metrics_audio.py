from __future__ import annotations

import httpx

from openinfer.metrics import Metrics
from openinfer.server import create_app
from tests.conftest import make_config


async def test_metrics_endpoint_counts_requests(client: httpx.AsyncClient):
    await client.post(
        "/v1/chat/completions",
        json={"model": "fast", "messages": [{"role": "user", "content": "x"}]},
    )
    r = await client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert (
        'openinfer_requests_total{backend="echo",endpoint="chat",model="echo",status="ok"} 1.0'
        in text
    )
    assert "openinfer_tokens_total" in text
    assert "openinfer_request_latency_seconds_bucket" in text


async def test_metrics_disabled_returns_501():
    app = create_app(make_config(), metrics=Metrics(enabled=False))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/metrics")
    assert r.status_code == 501
    assert r.json()["error"]["code"] == "NotSupported"


async def test_transcription_without_whisper_is_501(client: httpx.AsyncClient):
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        pass
    else:
        return  # extra installed in this environment; behaviour covered manually
    r = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"RIFF....", "audio/wav")},
        data={"model": "whisper-1"},
    )
    assert r.status_code == 501
    assert "uv sync --extra whisper" in r.json()["error"]["message"]

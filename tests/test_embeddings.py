from __future__ import annotations

import base64
import math

import httpx

from openinfer.backends.echo import hashed_embedding


def test_hashed_embedding_is_deterministic_and_unit_length():
    a = hashed_embedding("the quick brown fox")
    b = hashed_embedding("the quick brown fox")
    assert a == b
    assert len(a) == 384
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)


def test_hashed_embedding_shares_direction_for_overlapping_text():
    base = hashed_embedding("invoice total due friday")
    near = hashed_embedding("invoice total due monday")
    far = hashed_embedding("penguins swim in antarctica")
    dot = lambda u, v: sum(x * y for x, y in zip(u, v, strict=True))  # noqa: E731
    assert dot(base, near) > dot(base, far)


def test_hashed_embedding_dimensions_and_empty_text():
    assert len(hashed_embedding("x", dims=64)) == 64
    vec = hashed_embedding("")
    assert len(vec) == 384
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-9)


async def test_embeddings_endpoint(client: httpx.AsyncClient):
    r = await client.post(
        "/v1/embeddings", json={"model": "embed", "input": "hello", "dimensions": 16}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["object"] == "embedding"
    assert len(body["data"][0]["embedding"]) == 16
    assert body["model"] == "echo-embed"
    assert body["usage"]["prompt_tokens"] == body["usage"]["total_tokens"]
    assert r.headers["x-openinfer-backend"] == "echo"


async def test_embeddings_batch_and_errors(client: httpx.AsyncClient):
    r = await client.post(
        "/v1/embeddings", json={"model": "echo/echo-embed", "input": ["a", "b", "c"]}
    )
    assert [d["index"] for d in r.json()["data"]] == [0, 1, 2]
    r = await client.post("/v1/embeddings", json={"model": "embed", "input": []})
    assert r.status_code == 400
    r = await client.post(
        "/v1/embeddings", json={"model": "embed", "input": "x", "encoding_format": "base64"}
    )
    assert r.status_code == 200
    raw = base64.b64decode(r.json()["data"][0]["embedding"])
    assert len(raw) == 384 * 4  # float32 little endian, what the OpenAI SDK decodes
    assert r.json()["data"][0]["embedding"] != hashed_embedding("x")
    r = await client.post(
        "/v1/embeddings", json={"model": "embed", "input": "x", "encoding_format": "hex"}
    )
    assert r.status_code == 400

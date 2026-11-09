"""OpenAI wire format: request models and helpers that build response dicts.

Requests are validated loosely (`extra="allow"`) because clients send fields we do not
model (and OpenAI compatible backends may understand them). Responses are built as plain
dicts shaped exactly like OpenAI's so the official SDKs parse them without complaint.
See docs/learn/03-concepts/openai-compatible-apis-and-sse.md.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from openinfer.types import ChatRequest, ChatResult, EmbeddingResult, RoutingHints, Usage

# Body fields we handle ourselves; everything else is forwarded as `extra`.
_HANDLED = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "seed",
    "tools",
    "tool_choice",
    "response_format",
    "stream",
    "stream_options",
    "n",
    "user",
    "metadata",
    "openinfer",
}


class RoutingHintsBody(BaseModel):
    """`{"openinfer": {...}}` in the request body. Headers of the same name also work."""

    tags: list[str] = Field(default_factory=list)
    max_cost_usd: float | None = None
    max_latency_ms: int | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = ""
    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None
    openinfer: RoutingHintsBody | None = None

    def to_internal(self, hints: RoutingHints) -> ChatRequest:
        stop = [self.stop] if isinstance(self.stop, str) else self.stop
        extra = {k: v for k, v in (self.model_extra or {}).items() if k not in _HANDLED}
        return ChatRequest(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens or self.max_completion_tokens,
            stop=stop,
            seed=self.seed,
            tools=self.tools,
            tool_choice=self.tool_choice,
            response_format=self.response_format,
            hints=hints,
            extra=extra,
        )

    def wants_stream_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.get("include_usage"))


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "embed"
    input: str | list[str] | list[int] | list[list[int]]
    dimensions: int | None = None
    encoding_format: str | None = "float"
    user: str | None = None
    metadata: dict[str, Any] | None = None
    openinfer: RoutingHintsBody | None = None

    def inputs(self) -> list[str]:
        if isinstance(self.input, str):
            return [self.input]
        # Token id arrays are accepted for compatibility and stringified; we have no tokenizer.
        return [x if isinstance(x, str) else json.dumps(x) for x in self.input]


# -- hint parsing ------------------------------------------------------------------------------


def parse_hints(
    headers: Any,
    body: RoutingHintsBody | None,
    metadata: dict[str, Any] | None,
) -> RoutingHints:
    """Merge routing hints from headers, the `openinfer` body object and OpenAI `metadata`.

    Supported:
      X-OpenInfer-Tags: private,batch
      X-OpenInfer-Max-Cost-USD: 0.001
      X-OpenInfer-Max-Latency-Ms: 800
      body.openinfer: {"tags": [...], "max_cost_usd": ..., "max_latency_ms": ...}
      body.metadata: {"tags": "private" | ["private"], "privacy": "private"}
    """
    tags: set[str] = set()
    max_cost: float | None = None
    max_latency: int | None = None

    header_tags = headers.get("x-openinfer-tags") if headers is not None else None
    if header_tags:
        tags.update(t.strip() for t in header_tags.split(",") if t.strip())
    header_cost = headers.get("x-openinfer-max-cost-usd") if headers is not None else None
    if header_cost:
        max_cost = float(header_cost)
    header_latency = headers.get("x-openinfer-max-latency-ms") if headers is not None else None
    if header_latency:
        max_latency = int(float(header_latency))

    if body is not None:
        tags.update(body.tags)
        if body.max_cost_usd is not None:
            max_cost = _min_opt(max_cost, body.max_cost_usd)
        if body.max_latency_ms is not None:
            max_latency = _min_opt(max_latency, body.max_latency_ms)

    if metadata:
        meta_tags = metadata.get("tags")
        if isinstance(meta_tags, str):
            tags.update(t.strip() for t in meta_tags.split(",") if t.strip())
        elif isinstance(meta_tags, list):
            tags.update(str(t) for t in meta_tags)
        if metadata.get("privacy"):
            tags.add(str(metadata["privacy"]))

    return RoutingHints(tags=frozenset(tags), max_cost_usd=max_cost, max_latency_ms=max_latency)


def _min_opt(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


# -- response builders ---------------------------------------------------------------------------


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def route_info(entry_or_backend: Any, model: str, **extra: Any) -> dict[str, Any]:
    """The `openinfer` object we add to responses so clients can see what served them."""
    info: dict[str, Any] = {"backend": entry_or_backend, "model": model}
    info.update({k: v for k, v in extra.items() if v is not None})
    return info


def chat_completion_response(
    result: ChatResult,
    *,
    completion_id: str,
    created: int,
    info: dict[str, Any],
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": result.model,
        "system_fingerprint": "openinfer",
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": result.usage.to_openai(),
        "openinfer": info,
    }


def chat_chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: Usage | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": "openinfer",
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage.to_openai()
    if info is not None:
        chunk["openinfer"] = info
    return chunk


def usage_only_chunk(
    *, completion_id: str, created: int, model: str, usage: Usage
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": "openinfer",
        "choices": [],
        "usage": usage.to_openai(),
    }


def embeddings_response(result: EmbeddingResult, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(result.vectors)
        ],
        "model": result.model,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "total_tokens": result.usage.prompt_tokens,
        },
        "openinfer": info,
    }


def sse(data: dict[str, Any] | str) -> str:
    """Encode one server-sent event the way OpenAI does: `data: <json>\\n\\n`."""
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    return f"data: {payload}\n\n"


def now() -> int:
    return int(time.time())

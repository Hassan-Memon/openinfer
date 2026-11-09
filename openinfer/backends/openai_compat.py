"""Shared adapter for any server that speaks the OpenAI REST dialect.

vLLM, llama.cpp's server, LM Studio, Groq, OpenRouter, Gemini's compatibility endpoint and
OpenAI itself all accept the same `/chat/completions` JSON and stream the same
`data: {...}` server-sent events. This module implements that once; `cloud.py` and
`vllm.py` only add presets and metadata on top.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from openinfer.backends.base import Backend
from openinfer.config import BackendConfig
from openinfer.errors import (
    BackendError,
    BackendTimeout,
    BackendUnavailable,
    RateLimited,
)
from openinfer.types import (
    ChatChunk,
    ChatRequest,
    ChatResult,
    EmbeddingRequest,
    EmbeddingResult,
    HealthStatus,
    Usage,
    estimate_tokens,
)


class OpenAICompatBackend(Backend):
    kind = "openai-compat"
    local = False
    #: whether the server honours `stream_options.include_usage`
    supports_stream_usage: bool = True

    def __init__(
        self,
        name: str,
        config: BackendConfig,
        *,
        base_url: str,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(name, config)
        self.base_url = base_url.rstrip("/")
        merged = {"Content-Type": "application/json", **(headers or {}), **config.extra_headers}
        if api_key:
            merged["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=merged,
            timeout=httpx.Timeout(config.timeout_s, connect=10.0),
            transport=transport,
        )

    # -- request building ----------------------------------------------------------------

    def build_body(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"model": request.model, "messages": request.messages}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.stop:
            body["stop"] = request.stop
        if request.seed is not None:
            body["seed"] = request.seed
        if request.tools:
            body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.response_format:
            body["response_format"] = request.response_format
        for key, value in request.extra.items():
            body.setdefault(key, value)
        if stream:
            body["stream"] = True
            if self.supports_stream_usage:
                body["stream_options"] = {"include_usage": True}
        return body

    # -- error mapping -------------------------------------------------------------------

    def _raise_for_status(self, response: httpx.Response, body_text: str = "") -> None:
        status = response.status_code
        if status < 400:
            return
        detail = _error_message(body_text or response.text) or f"HTTP {status}"
        if status == 429:
            raise RateLimited(detail, backend=self.name)
        if status >= 500:
            raise BackendError(detail, backend=self.name, status_code=502, retriable=True)
        raise BackendError(detail, backend=self.name, status_code=502, retriable=False)

    def _translate(self, exc: Exception) -> BackendError:
        if isinstance(exc, BackendError):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return BackendTimeout(f"timed out after {self.config.timeout_s}s", backend=self.name)
        if isinstance(exc, httpx.ConnectError | httpx.NetworkError):
            return BackendUnavailable(f"cannot reach {self.base_url}: {exc}", backend=self.name)
        if isinstance(exc, httpx.HTTPError):
            return BackendError(str(exc), backend=self.name, retriable=False)
        return BackendError(f"{type(exc).__name__}: {exc}", backend=self.name, retriable=False)

    # -- chat ------------------------------------------------------------------------------

    async def chat(self, request: ChatRequest) -> ChatResult:
        try:
            response = await self._client.post(
                "/chat/completions", json=self.build_body(request, stream=False)
            )
            self._raise_for_status(response)
            data = response.json()
        except Exception as exc:
            raise self._translate(exc) from exc
        return parse_chat_completion(data, request)

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        body = self.build_body(request, stream=True)
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    text = (await response.aread()).decode("utf-8", "replace")
                    self._raise_for_status(response, text)
                async for payload in iter_sse(response):
                    chunk = parse_chunk(payload)
                    if chunk is not None:
                        yield chunk
        except Exception as exc:
            raise self._translate(exc) from exc

    # -- embeddings -------------------------------------------------------------------------

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        body: dict[str, Any] = {"model": request.model, "input": request.inputs}
        if request.dimensions:
            body["dimensions"] = request.dimensions
        try:
            response = await self._client.post("/embeddings", json=body)
            self._raise_for_status(response)
            data = response.json()
        except Exception as exc:
            raise self._translate(exc) from exc
        rows = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
        usage = _parse_usage(data.get("usage"), request.prompt_text(), "")
        return EmbeddingResult(
            vectors=[row["embedding"] for row in rows],
            usage=usage,
            model=data.get("model") or request.model,
        )

    # -- housekeeping -----------------------------------------------------------------------

    async def health(self) -> HealthStatus:
        try:
            response = await self._client.get("/models", timeout=3.0)
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(ok=False, detail=f"{type(exc).__name__}: {exc}")
        if response.status_code >= 400:
            return HealthStatus(ok=False, detail=f"HTTP {response.status_code}")
        return HealthStatus(ok=True, detail=f"{len(response.json().get('data', []))} models")

    async def aclose(self) -> None:
        await self._client.aclose()


# -- parsing helpers (module level so Ollama and llama.cpp can reuse them) ---------------------


def parse_chat_completion(data: dict[str, Any], request: ChatRequest) -> ChatResult:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    usage = _parse_usage(data.get("usage"), request.prompt_text(), content or "")
    return ChatResult(
        content=content,
        reasoning=message.get("reasoning_content") or message.get("reasoning") or None,
        tool_calls=message.get("tool_calls") or None,
        finish_reason=choice.get("finish_reason") or "stop",
        usage=usage,
        model=data.get("model") or request.model,
    )


def parse_chunk(payload: dict[str, Any]) -> ChatChunk | None:
    choices = payload.get("choices") or []
    usage = payload.get("usage")
    parsed_usage = (
        Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), exact=True)
        if usage
        else None
    )
    if not choices:
        # Final usage-only chunk (stream_options.include_usage)
        return ChatChunk(usage=parsed_usage) if parsed_usage else None
    choice = choices[0]
    delta = choice.get("delta") or {}
    return ChatChunk(
        delta=delta.get("content"),
        reasoning=delta.get("reasoning_content") or delta.get("reasoning") or None,
        tool_calls=delta.get("tool_calls") or None,
        finish_reason=choice.get("finish_reason"),
        usage=parsed_usage,
    )


async def iter_sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded JSON objects from an OpenAI style `text/event-stream` body."""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _parse_usage(raw: dict[str, Any] | None, prompt_text: str, completion_text: str) -> Usage:
    if raw and ("prompt_tokens" in raw or "completion_tokens" in raw):
        return Usage(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            exact=True,
        )
    return Usage(estimate_tokens(prompt_text), estimate_tokens(completion_text), exact=False)


def _error_message(text: str) -> str | None:
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:300]
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err)
    if isinstance(err, str):
        return err
    return text[:300]

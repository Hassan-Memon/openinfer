"""Ollama backend using Ollama's native HTTP API.

Ollama also exposes an OpenAI compatible `/v1`, but the native `/api/chat` endpoint reports
exact prompt and completion token counts (`prompt_eval_count`, `eval_count`) and timing, which
the ledger and bench want. Streaming is newline delimited JSON rather than SSE.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from openinfer.backends.base import Backend
from openinfer.config import BackendConfig
from openinfer.errors import BackendError, BackendTimeout, BackendUnavailable, RateLimited
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


class OllamaBackend(Backend):
    kind = "ollama"
    local = True

    def __init__(
        self,
        name: str,
        config: BackendConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(name, config)
        self.base_url = (config.base_url or "http://localhost:11434").rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=config.extra_headers,
            timeout=httpx.Timeout(config.timeout_s, connect=5.0),
            transport=transport,
        )

    def _body(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.stop:
            options["stop"] = request.stop
        if request.seed is not None:
            options["seed"] = request.seed
        body: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "stream": stream,
            "options": options,
        }
        if request.tools:
            body["tools"] = request.tools
        if request.response_format and request.response_format.get("type") == "json_object":
            body["format"] = "json"
        return body

    def _translate(self, exc: Exception) -> BackendError:
        if isinstance(exc, BackendError):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return BackendTimeout(f"timed out after {self.config.timeout_s}s", backend=self.name)
        if isinstance(exc, httpx.ConnectError | httpx.NetworkError):
            return BackendUnavailable(f"cannot reach {self.base_url}: {exc}", backend=self.name)
        return BackendError(f"{type(exc).__name__}: {exc}", backend=self.name, retriable=False)

    def _raise_for_status(self, response: httpx.Response, text: str = "") -> None:
        status = response.status_code
        if status < 400:
            return
        text = text or response.text
        try:
            detail = json.loads(text).get("error") or text
        except (json.JSONDecodeError, AttributeError):
            detail = text[:300] or f"HTTP {status}"
        if status == 429:
            raise RateLimited(str(detail), backend=self.name)
        if status == 404 and "not found" in str(detail):
            raise BackendError(
                f"{detail} (run `ollama pull <model>`)", backend=self.name, retriable=False
            )
        raise BackendError(str(detail), backend=self.name, retriable=status >= 500)

    async def chat(self, request: ChatRequest) -> ChatResult:
        try:
            response = await self._client.post("/api/chat", json=self._body(request, stream=False))
            self._raise_for_status(response)
            data = response.json()
        except Exception as exc:
            raise self._translate(exc) from exc
        message = data.get("message") or {}
        content = message.get("content")
        tool_calls = _convert_tool_calls(message.get("tool_calls"))
        return ChatResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else _finish(data.get("done_reason")),
            usage=_usage(data, request.prompt_text(), content or ""),
            model=data.get("model") or request.model,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        try:
            async with self._client.stream(
                "POST", "/api/chat", json=self._body(request, stream=True)
            ) as response:
                if response.status_code >= 400:
                    text = (await response.aread()).decode("utf-8", "replace")
                    self._raise_for_status(response, text)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    message = data.get("message") or {}
                    if data.get("done"):
                        yield ChatChunk(
                            delta=message.get("content") or None,
                            finish_reason=_finish(data.get("done_reason")),
                            usage=_usage(data, request.prompt_text(), ""),
                        )
                        return
                    yield ChatChunk(
                        delta=message.get("content") or None,
                        tool_calls=_convert_tool_calls(message.get("tool_calls")),
                    )
        except Exception as exc:
            raise self._translate(exc) from exc

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        try:
            response = await self._client.post(
                "/api/embed", json={"model": request.model, "input": request.inputs}
            )
            self._raise_for_status(response)
            data = response.json()
        except Exception as exc:
            raise self._translate(exc) from exc
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        usage = (
            Usage(prompt_tokens=prompt_tokens, exact=True)
            if prompt_tokens
            else Usage(prompt_tokens=estimate_tokens(request.prompt_text()), exact=False)
        )
        return EmbeddingResult(
            vectors=data.get("embeddings") or [],
            usage=usage,
            model=data.get("model") or request.model,
        )

    async def health(self) -> HealthStatus:
        try:
            response = await self._client.get("/api/tags", timeout=3.0)
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(ok=False, detail=f"{type(exc).__name__}: {exc}")
        if response.status_code >= 400:
            return HealthStatus(ok=False, detail=f"HTTP {response.status_code}")
        names = [m.get("name") for m in response.json().get("models", [])]
        return HealthStatus(ok=True, detail=f"{len(names)} models pulled")

    async def list_pulled_models(self) -> list[str]:
        response = await self._client.get("/api/tags", timeout=3.0)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    async def aclose(self) -> None:
        await self._client.aclose()


def _finish(done_reason: str | None) -> str:
    if done_reason == "length":
        return "length"
    return "stop"


def _usage(data: dict[str, Any], prompt_text: str, completion_text: str) -> Usage:
    prompt = data.get("prompt_eval_count")
    completion = data.get("eval_count")
    if prompt is not None or completion is not None:
        return Usage(int(prompt or 0), int(completion or 0), exact=True)
    return Usage(estimate_tokens(prompt_text), estimate_tokens(completion_text), exact=False)


def _convert_tool_calls(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Ollama returns `function.arguments` as an object; OpenAI clients expect a JSON string."""
    if not raw:
        return None
    converted = []
    for i, call in enumerate(raw):
        fn = call.get("function") or {}
        args = fn.get("arguments", {})
        converted.append(
            {
                "id": call.get("id") or f"call_{i}",
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            }
        )
    return converted

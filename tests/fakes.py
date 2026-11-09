"""Scriptable fake backend for router tests.

`script` is a list consumed one item per call: `None` means succeed, an exception instance
means raise it. When the script runs out the backend keeps succeeding.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openinfer.backends.base import Backend
from openinfer.config import BackendConfig
from openinfer.types import (
    ChatChunk,
    ChatRequest,
    ChatResult,
    EmbeddingRequest,
    EmbeddingResult,
    Usage,
)


class FakeBackend(Backend):
    kind = "fake"

    def __init__(
        self,
        name: str,
        *,
        local: bool = True,
        models: list[str] | None = None,
        embedding_models: list[str] | None = None,
        script: list[Exception | None] | None = None,
        pricing_provider: str | None = None,
        expected_latency_ms: int | None = None,
        stream_words: int = 3,
    ) -> None:
        cfg = BackendConfig(
            type="echo",
            models=models or [f"{name}-model"],
            embedding_models=embedding_models or [],
            expected_latency_ms=expected_latency_ms,
        )
        super().__init__(name, cfg)
        self.local = local
        self.pricing_provider = pricing_provider
        self.script = list(script or [])
        self.calls: list[tuple[str, Any]] = []
        self.stream_words = stream_words

    def _next(self) -> None:
        if self.script:
            item = self.script.pop(0)
            if item is not None:
                raise item

    async def chat(self, request: ChatRequest) -> ChatResult:
        self.calls.append(("chat", request.model))
        self._next()
        return ChatResult(
            content=f"reply from {self.name}",
            usage=Usage(prompt_tokens=10, completion_tokens=20),
            model=request.model,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.calls.append(("stream", request.model))
        self._next()
        for i in range(self.stream_words):
            yield ChatChunk(delta=f"w{i} ")
        yield ChatChunk(finish_reason="stop", usage=Usage(prompt_tokens=5, completion_tokens=3))

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.calls.append(("embed", request.model))
        self._next()
        return EmbeddingResult(
            vectors=[[1.0, 0.0] for _ in request.inputs],
            usage=Usage(prompt_tokens=len(request.inputs)),
            model=request.model,
        )

"""The adapter interface every backend implements.

The router only talks to this class. A backend is responsible for:
- translating `ChatRequest` / `EmbeddingRequest` into whatever its engine wants
- translating the engine's answer back into `ChatResult` / `ChatChunk` / `EmbeddingResult`
- raising `BackendError` subclasses with the right `retriable` flag

It must not know about aliases, policies, pricing or the ledger.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from openinfer.config import BackendConfig
from openinfer.errors import NotSupported
from openinfer.types import (
    ChatChunk,
    ChatRequest,
    ChatResult,
    EmbeddingRequest,
    EmbeddingResult,
    HealthStatus,
)


class Backend(ABC):
    #: short identifier of the adapter kind: echo, ollama, vllm, cloud, llamacpp
    kind: str = "base"
    #: True when inference happens on hardware you control (no data leaves the machine/network)
    local: bool = True
    #: pricing table provider (groq, openai, gemini, openrouter) or None for local engines
    pricing_provider: str | None = None

    def __init__(self, name: str, config: BackendConfig) -> None:
        self.name = name
        self.config = config
        self.models: list[str] = list(config.models)
        self.embedding_models: list[str] = list(config.embedding_models)
        self.expected_latency_ms: int | None = config.expected_latency_ms

    # -- capabilities -----------------------------------------------------------------

    def serves(self, model: str) -> bool:
        return model in self.models or model in self.embedding_models

    def default_chat_model(self) -> str | None:
        return self.models[0] if self.models else None

    # -- inference ---------------------------------------------------------------------

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResult: ...

    @abstractmethod
    def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]: ...

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        raise NotSupported(f"backend {self.name!r} does not support embeddings")

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True)

    async def aclose(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name} local={self.local}>"

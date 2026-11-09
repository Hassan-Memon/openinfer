"""Echo backend: deterministic, offline, zero cost.

It exists so that tests, CI, `docker compose up` with no keys and the Hugging Face demo all
work without a model. It mirrors the last user message, streams it word by word, can emit a
tool call when the caller forces one, and returns hashed embeddings that are stable across
runs (the same text always gives the same vector).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
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
    estimate_tokens,
    message_text,
)

DEFAULT_DIMENSIONS = 384
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class EchoBackend(Backend):
    kind = "echo"
    local = True

    def __init__(self, name: str, config: BackendConfig) -> None:
        super().__init__(name, config)
        if not self.models:
            self.models = ["echo"]
        if not self.embedding_models:
            self.embedding_models = ["echo-embed"]
        self.delay_ms: float = float(config.options.get("delay_ms", 0))

    # -- chat ----------------------------------------------------------------------------

    def _reply(self, request: ChatRequest) -> tuple[str | None, list[dict[str, Any]] | None]:
        forced = _forced_tool(request)
        if forced is not None:
            call = {
                "id": "call_echo_0",
                "type": "function",
                "function": {"name": forced, "arguments": json.dumps({})},
            }
            return None, [call]
        last_user = next((m for m in reversed(request.messages) if m.get("role") == "user"), None)
        text = message_text(last_user) if last_user else ""
        return f"Echo: {text}" if text else "Echo: (empty message)", None

    def _usage(self, request: ChatRequest, completion: str | None) -> Usage:
        return Usage(
            prompt_tokens=estimate_tokens(request.prompt_text()),
            completion_tokens=len(_TOKEN_RE.findall(completion or "")),
            exact=False,
        )

    async def chat(self, request: ChatRequest) -> ChatResult:
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)
        content, tool_calls = self._reply(request)
        return ChatResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=self._usage(request, content),
            model=request.model,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        content, tool_calls = self._reply(request)
        if tool_calls:
            yield ChatChunk(tool_calls=tool_calls)
            yield ChatChunk(finish_reason="tool_calls", usage=self._usage(request, None))
            return
        assert content is not None
        words = content.split(" ")
        for i, word in enumerate(words):
            if self.delay_ms:
                await asyncio.sleep(self.delay_ms / 1000)
            yield ChatChunk(delta=word if i == len(words) - 1 else word + " ")
        yield ChatChunk(finish_reason="stop", usage=self._usage(request, content))

    # -- embeddings -----------------------------------------------------------------------

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        dims = request.dimensions or DEFAULT_DIMENSIONS
        vectors = [hashed_embedding(text, dims) for text in request.inputs]
        usage = Usage(prompt_tokens=estimate_tokens(request.prompt_text()), exact=False)
        return EmbeddingResult(vectors=vectors, usage=usage, model=request.model)


def _forced_tool(request: ChatRequest) -> str | None:
    """Return a tool name when the caller demands a tool call, else None."""
    if not request.tools:
        return None
    choice = request.tool_choice
    if isinstance(choice, dict):
        fn = choice.get("function") or {}
        return fn.get("name") or None
    if choice == "required":
        first = request.tools[0]
        return (first.get("function") or {}).get("name")
    return None


def hashed_embedding(text: str, dims: int = DEFAULT_DIMENSIONS) -> list[float]:
    """Feature hashing: each token bumps one dimension chosen by its hash, sign by another bit.

    Not semantically meaningful, but deterministic, unit length, and texts that share words
    share directions, which is enough for demos and tests of the vector plumbing.
    """
    vec = [0.0] * dims
    tokens = _TOKEN_RE.findall(text.lower()) or [""]
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dims
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

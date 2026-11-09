"""Internal request and result types.

These are deliberately *not* the OpenAI wire format. The server translates OpenAI JSON into
these dataclasses, the router and backends only ever see these, and the server translates
results back. That keeps provider quirks at the edges: a backend adapter is the only place
that knows what Ollama or Groq actually send over the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class RoutingHints:
    """Per request routing constraints. Parsed from headers or the `openinfer` body object."""

    tags: frozenset[str] = frozenset()
    max_cost_usd: float | None = None
    max_latency_ms: int | None = None

    def has(self, tag: str) -> bool:
        return tag in self.tags


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # True when the backend reported counts; False when we estimated them from text length.
    exact: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_openai(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class ChatRequest:
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    hints: RoutingHints = field(default_factory=RoutingHints)
    # Anything else from the OpenAI body that an OpenAI compatible backend may understand.
    extra: dict[str, Any] = field(default_factory=dict)

    def with_model(self, model: str) -> ChatRequest:
        """Copy of the request pointed at a concrete model name a backend understands."""
        return ChatRequest(
            model=model,
            messages=self.messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop,
            seed=self.seed,
            tools=self.tools,
            tool_choice=self.tool_choice,
            response_format=self.response_format,
            hints=self.hints,
            extra=self.extra,
        )

    def prompt_text(self) -> str:
        return "\n".join(message_text(m) for m in self.messages)


@dataclass(slots=True)
class ChatResult:
    content: str | None
    usage: Usage
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    model: str = ""
    backend: str = ""


@dataclass(slots=True)
class ChatChunk:
    """One streamed delta. The last chunk carries `finish_reason` and, if known, `usage`."""

    delta: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: Usage | None = None


@dataclass(slots=True)
class EmbeddingRequest:
    model: str
    inputs: list[str]
    dimensions: int | None = None
    hints: RoutingHints = field(default_factory=RoutingHints)

    def with_model(self, model: str) -> EmbeddingRequest:
        return EmbeddingRequest(
            model=model, inputs=self.inputs, dimensions=self.dimensions, hints=self.hints
        )

    def prompt_text(self) -> str:
        return "\n".join(self.inputs)


@dataclass(slots=True)
class EmbeddingResult:
    vectors: list[list[float]]
    usage: Usage
    model: str = ""
    backend: str = ""


@dataclass(slots=True)
class HealthStatus:
    ok: bool
    detail: str = ""


def message_text(message: dict[str, Any]) -> str:
    """Flatten an OpenAI message's content (string or list of parts) to plain text."""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif isinstance(part, str):
            parts.append(part)
    return "".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token estimate used when a backend does not report usage.

    English text averages roughly four characters per token for BPE tokenizers. We never
    bill from this number without flagging `Usage.exact=False`.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class LedgerEntry:
    endpoint: str  # chat, chat_stream, embeddings, transcription
    requested_model: str
    backend: str
    model: str
    status: str = "ok"  # ok | error
    alias: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_exact: bool = True
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    attempts: int = 1
    tags: str = ""
    error: str | None = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Ledger(ABC):
    @abstractmethod
    def record(self, entry: LedgerEntry) -> None: ...

    @abstractmethod
    def recent(self, limit: int = 50) -> list[dict[str, Any]]: ...

    @abstractmethod
    def stats(self, hours: float = 24.0) -> dict[str, Any]: ...

    @abstractmethod
    def latency_p50(self, backend: str, model: str, *, min_samples: int = 5) -> float | None:
        """Median latency of recent successful calls, or None if fewer than min_samples."""

    def close(self) -> None:
        return None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)

"""Prometheus metrics (optional extra `metrics`).

When `prometheus_client` is missing every method is a no-op and `/metrics` answers 501. Each
app gets its own registry so tests can build many apps in one process without
"duplicate timeseries" errors. See docs/learn/03-concepts/prometheus-metrics.md.
"""

from __future__ import annotations

from typing import Any

from openinfer.ledger.base import LedgerEntry

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    PROMETHEUS_AVAILABLE = False

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)


class Metrics:
    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = (
            PROMETHEUS_AVAILABLE if enabled is None else (enabled and PROMETHEUS_AVAILABLE)
        )
        if not self.enabled:
            return
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "openinfer_requests_total",
            "Requests routed, by backend, model, endpoint and status",
            ["backend", "model", "endpoint", "status"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "openinfer_tokens_total",
            "Tokens processed, by backend, model and direction (prompt|completion)",
            ["backend", "model", "direction"],
            registry=self.registry,
        )
        self.cost = Counter(
            "openinfer_cost_usd_total",
            "Estimated spend in USD, by backend and model",
            ["backend", "model"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "openinfer_request_latency_seconds",
            "End to end request latency",
            ["backend", "endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.ttft = Histogram(
            "openinfer_ttft_seconds",
            "Time to first token for streamed requests",
            ["backend"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "openinfer_fallbacks_total",
            "Requests that needed more than one attempt",
            ["requested_model"],
            registry=self.registry,
        )

    def observe(self, entry: LedgerEntry) -> None:
        if not self.enabled:
            return
        self.requests.labels(entry.backend, entry.model, entry.endpoint, entry.status).inc()
        if entry.status == "ok":
            self.tokens.labels(entry.backend, entry.model, "prompt").inc(entry.prompt_tokens)
            self.tokens.labels(entry.backend, entry.model, "completion").inc(
                entry.completion_tokens
            )
            self.cost.labels(entry.backend, entry.model).inc(entry.cost_usd)
            self.latency.labels(entry.backend, entry.endpoint).observe(entry.latency_ms / 1000)
            if entry.ttft_ms is not None:
                self.ttft.labels(entry.backend).observe(entry.ttft_ms / 1000)
        if entry.attempts > 1:
            self.fallbacks.labels(entry.requested_model).inc()

    def render(self) -> tuple[bytes, str]:
        if not self.enabled:
            raise RuntimeError("prometheus_client not installed")
        return generate_latest(self.registry), CONTENT_TYPE_LATEST

    def snapshot(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

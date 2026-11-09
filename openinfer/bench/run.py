"""`openinfer bench`: tokens per second, time to first token and cost across backends.

Each backend's first chat model is streamed `runs` times with the same prompt. We measure
TTFT (time until the first content delta) and generation throughput
(completion tokens / time after first token). Cost per million output tokens comes from the
pricing table (or an override in openinfer.yaml). The echo backend makes the command work
offline so the pipeline is testable; the numbers only mean something on real engines.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from openinfer.backends.base import Backend
from openinfer.config import AppConfig
from openinfer.errors import OpenInferError
from openinfer.ledger.base import percentile
from openinfer.pricing import Pricing
from openinfer.types import ChatRequest, estimate_tokens

DEFAULT_PROMPT = (
    "Explain in three short paragraphs why quantised models run well on consumer CPUs. "
    "Use plain English."
)


@dataclass(slots=True)
class BenchResult:
    backend: str
    model: str
    local: bool
    runs: int = 0
    ttft_ms_p50: float | None = None
    tokens_per_s: float | None = None
    total_ms_p50: float | None = None
    completion_tokens_avg: float | None = None
    usd_per_1m_output: float | None = None
    usd_per_1m_input: float | None = None
    price_known: bool = True
    status: str = "ok"
    error: str | None = None
    samples: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("samples", None)
        return data


async def bench_backend(
    backend: Backend,
    model: str,
    *,
    prompt: str,
    runs: int,
    max_tokens: int,
    pricing: Pricing,
) -> BenchResult:
    price = pricing.price(
        backend=backend.name, provider=backend.pricing_provider, model=model, local=backend.local
    )
    result = BenchResult(
        backend=backend.name,
        model=model,
        local=backend.local,
        usd_per_1m_output=price.output_per_1m,
        usd_per_1m_input=price.input_per_1m,
        price_known=price.known,
    )
    request = ChatRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    ttfts: list[float] = []
    totals: list[float] = []
    rates: list[float] = []
    tokens: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        completion_tokens: int | None = None
        try:
            async for chunk in backend.chat_stream(request):
                text = (chunk.delta or "") + (chunk.reasoning or "")
                if text and first is None:
                    first = time.perf_counter()
                if text:
                    parts.append(text)
                if chunk.usage is not None:
                    completion_tokens = chunk.usage.completion_tokens
        except OpenInferError as exc:
            result.status = "error"
            result.error = str(exc)[:200]
            break
        end = time.perf_counter()
        if first is None:
            first = end
        if completion_tokens is None:
            completion_tokens = estimate_tokens("".join(parts))
        gen_seconds = max(end - first, 1e-6)
        ttfts.append((first - started) * 1000)
        totals.append((end - started) * 1000)
        rates.append(completion_tokens / gen_seconds)
        tokens.append(float(completion_tokens))
        result.samples.append(
            {
                "ttft_ms": ttfts[-1],
                "total_ms": totals[-1],
                "tokens_per_s": rates[-1],
                "completion_tokens": completion_tokens,
            }
        )
    result.runs = len(totals)
    if totals:
        result.ttft_ms_p50 = percentile(ttfts, 0.5)
        result.total_ms_p50 = percentile(totals, 0.5)
        result.tokens_per_s = percentile(rates, 0.5)
        result.completion_tokens_avg = sum(tokens) / len(tokens)
    return result


async def run_bench(
    config: AppConfig,
    backends: dict[str, Backend],
    *,
    only: list[str] | None = None,
    prompt: str = DEFAULT_PROMPT,
    runs: int = 3,
    max_tokens: int = 128,
    pricing: Pricing | None = None,
    skip_unhealthy: bool = True,
) -> list[BenchResult]:
    pricing = pricing or Pricing.load_default()
    for name, backend in backends.items():
        pricing.add_overrides(name, backend.config.pricing)
    selected = {n: b for n, b in backends.items() if not only or n in only}
    results: list[BenchResult] = []
    for name, backend in selected.items():
        model = backend.default_chat_model()
        if model is None:
            continue
        if skip_unhealthy:
            try:
                status = await asyncio.wait_for(backend.health(), timeout=3.0)
            except Exception as exc:  # noqa: BLE001 - report instead of abort
                status = type("S", (), {"ok": False, "detail": str(exc)})()
            if not status.ok:
                results.append(
                    BenchResult(
                        backend=name,
                        model=model,
                        local=backend.local,
                        status="unreachable",
                        error=status.detail[:200],
                        price_known=True,
                        usd_per_1m_output=0.0 if backend.local else None,
                    )
                )
                continue
        results.append(
            await bench_backend(
                backend, model, prompt=prompt, runs=runs, max_tokens=max_tokens, pricing=pricing
            )
        )
    return results


def markdown_table(results: list[BenchResult]) -> str:
    header = (
        "| Backend | Model | Where | TTFT p50 (ms) | Gen tokens/s | Total p50 (ms) "
        "| $/1M out tokens | Status |\n"
        "|---|---|---|---:|---:|---:|---:|---|"
    )
    rows = []
    for r in results:
        price = (
            "n/a"
            if r.usd_per_1m_output is None
            else f"{r.usd_per_1m_output:.2f}" + ("" if r.price_known else " (unknown, assumed 0)")
        )
        rows.append(
            "| {b} | {m} | {w} | {t} | {r} | {tot} | {p} | {s} |".format(
                b=r.backend,
                m=r.model,
                w="local" if r.local else "cloud",
                t=_fmt(r.ttft_ms_p50),
                r=_fmt(r.tokens_per_s),
                tot=_fmt(r.total_ms_p50),
                p=price,
                s=r.status if not r.error else f"{r.status}: {r.error}",
            )
        )
    return "\n".join([header, *rows])


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:,.1f}"

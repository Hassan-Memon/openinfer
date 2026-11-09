"""The policy engine.

A request names a model. The router:

1. resolves that name to an ordered list of (backend, model) candidates
   (alias chain, "backend/model", or any backend that lists the bare model name)
2. drops candidates the request's constraints forbid:
   - privacy tag: cloud backends are removed, fail closed if nothing local remains
   - cost ceiling: estimated cost (prompt tokens + max_tokens at list price) must fit
   - latency ceiling: measured p50 from the ledger, else the configured expectation
3. tries the survivors in order, retrying transient failures on the same backend with
   exponential backoff, then moving to the next candidate
4. records one ledger row (and Prometheus samples) per request

Streaming follows the same path, but fallback is only possible before the first token has
been sent to the client. After that, an error terminates the stream.

docs/learn/03-concepts/routing-policies-and-fallbacks.md walks through this file.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from openinfer.backends.base import Backend
from openinfer.config import AppConfig, PolicyConfig
from openinfer.errors import (
    AllBackendsFailed,
    BackendError,
    ModelNotFound,
    NoCandidates,
)
from openinfer.ledger.base import Ledger, LedgerEntry
from openinfer.metrics import Metrics
from openinfer.pricing import Pricing
from openinfer.types import (
    ChatChunk,
    ChatRequest,
    ChatResult,
    EmbeddingRequest,
    EmbeddingResult,
    HealthStatus,
    RoutingHints,
    Usage,
    estimate_tokens,
)

log = logging.getLogger("openinfer.router")

T = TypeVar("T")
Sleeper = Callable[[float], Awaitable[None]]


@dataclass(slots=True, frozen=True)
class Candidate:
    backend: Backend
    model: str

    @property
    def qualified(self) -> str:
        return f"{self.backend.name}/{self.model}"


@dataclass(slots=True)
class Plan:
    requested: str
    alias: str | None
    candidates: list[Candidate]
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (qualified, reason)


class Router:
    def __init__(
        self,
        config: AppConfig,
        backends: dict[str, Backend],
        ledger: Ledger,
        pricing: Pricing | None = None,
        *,
        metrics: Metrics | None = None,
        sleep: Sleeper = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.policy: PolicyConfig = config.policy
        self.backends = backends
        self.ledger = ledger
        self.pricing = pricing or Pricing.load_default()
        self.metrics = metrics or Metrics(enabled=False)
        self._sleep = sleep
        self._rng = rng or random.Random()
        for name, backend in backends.items():
            self.pricing.add_overrides(name, backend.config.pricing)

    # -- resolution ------------------------------------------------------------------------

    def resolve(self, model: str) -> tuple[str | None, list[Candidate]]:
        """Turn a requested model name into (alias or None, ordered candidates)."""
        name = (model or "").strip() or self.policy.default_alias
        if name in self.config.aliases:
            candidates = []
            for entry in self.config.aliases[name]:
                backend_name, _, concrete = entry.partition("/")
                backend = self.backends.get(backend_name)
                if backend is None or not concrete:
                    continue  # disabled, unkeyed, or malformed entry: silently skipped
                candidates.append(Candidate(backend, concrete))
            if not candidates:
                raise ModelNotFound(
                    f"alias {name!r} has no enabled backends; check API keys and `enabled` flags"
                )
            return name, candidates
        backend_name, slash, concrete = name.partition("/")
        if slash and backend_name in self.backends and concrete:
            return None, [Candidate(self.backends[backend_name], concrete)]
        matches = [Candidate(b, name) for b in self.backends.values() if b.serves(name)]
        if matches:
            return None, matches
        aliases = ", ".join(sorted(self.config.aliases)) or "none"
        raise ModelNotFound(
            f"model {name!r} is not an alias ({aliases}), not 'backend/model', and no enabled "
            "backend lists it. See GET /v1/models."
        )

    # -- planning ----------------------------------------------------------------------------

    def estimate_cost(self, cand: Candidate, prompt_tokens: int, max_tokens: int) -> float:
        price = self.pricing.price(
            backend=cand.backend.name,
            provider=cand.backend.pricing_provider,
            model=cand.model,
            local=cand.backend.local,
        )
        return price.cost(prompt_tokens, max_tokens)

    def expected_latency_ms(self, cand: Candidate) -> float | None:
        measured = self.ledger.latency_p50(
            cand.backend.name, cand.model, min_samples=self.policy.latency_min_samples
        )
        if measured is not None:
            return measured
        return cand.backend.expected_latency_ms

    def plan(
        self,
        model: str,
        hints: RoutingHints,
        *,
        prompt_tokens: int = 0,
        max_tokens: int | None = None,
    ) -> Plan:
        alias, candidates = self.resolve(model)
        plan = Plan(requested=model or alias or "", alias=alias, candidates=[])

        private = hints.has(self.policy.private_tag) or alias == self.policy.private_tag
        cost_ceiling = _min_opt(hints.max_cost_usd, self.policy.max_cost_usd)
        latency_ceiling = _min_opt(hints.max_latency_ms, self.policy.max_latency_ms)
        out_tokens = max_tokens or self.policy.default_max_tokens

        for cand in candidates:
            if private and not cand.backend.local:
                plan.rejected.append((cand.qualified, "privacy: cloud backend"))
                continue
            if cost_ceiling is not None:
                est = self.estimate_cost(cand, prompt_tokens, out_tokens)
                if est > cost_ceiling:
                    plan.rejected.append(
                        (cand.qualified, f"cost: est ${est:.6f} > ceiling ${cost_ceiling:.6f}")
                    )
                    continue
            if latency_ceiling is not None:
                expected = self.expected_latency_ms(cand)
                if expected is not None and expected > latency_ceiling:
                    plan.rejected.append(
                        (
                            cand.qualified,
                            f"latency: expected {expected:.0f}ms > {latency_ceiling}ms",
                        )
                    )
                    continue
            plan.candidates.append(cand)

        if not plan.candidates:
            reasons = "; ".join(f"{q} ({r})" for q, r in plan.rejected) or "no candidates"
            if private and all("privacy" in r for _, r in plan.rejected):
                raise NoCandidates(
                    f"request is tagged {self.policy.private_tag!r} but every candidate for "
                    f"{plan.requested!r} is a cloud backend. Refusing to send data off-machine. "
                    f"Rejected: {reasons}"
                )
            raise NoCandidates(
                f"no candidate for {plan.requested!r} satisfies the policy: {reasons}"
            )
        return plan

    # -- execution ----------------------------------------------------------------------------

    def _backoff_seconds(self, attempt: int) -> float:
        base = self.policy.backoff_base_ms / 1000
        cap = self.policy.backoff_max_ms / 1000
        delay = min(cap, base * (2**attempt))
        return delay + self._rng.uniform(0, base / 2)

    async def _try_candidates(
        self,
        plan: Plan,
        call: Callable[[Candidate], Awaitable[T]],
    ) -> tuple[T, Candidate, int, list[str]]:
        """Run `call` against each candidate with per backend retries. Returns attempt count."""
        failures: list[str] = []
        attempts = 0
        for cand in plan.candidates:
            for retry in range(self.policy.retries + 1):
                attempts += 1
                try:
                    return await call(cand), cand, attempts, failures
                except BackendError as exc:
                    failures.append(f"{cand.qualified}: {exc.message}")
                    if not exc.retriable or retry >= self.policy.retries:
                        break
                    delay = self._backoff_seconds(retry)
                    log.info("retrying %s in %.2fs after: %s", cand.qualified, delay, exc.message)
                    await self._sleep(delay)
                except Exception as exc:
                    failures.append(f"{cand.qualified}: {type(exc).__name__}: {exc}")
                    log.exception("backend %s raised", cand.qualified)
                    break
        raise AllBackendsFailed(failures)

    def _entry(
        self,
        endpoint: str,
        plan: Plan,
        cand: Candidate | None,
        hints: RoutingHints,
        *,
        usage: Usage | None,
        latency_ms: float,
        ttft_ms: float | None = None,
        attempts: int,
        status: str = "ok",
        error: str | None = None,
    ) -> LedgerEntry:
        usage = usage or Usage(0, 0, exact=False)
        cost = 0.0
        if cand is not None and status == "ok":
            price = self.pricing.price(
                backend=cand.backend.name,
                provider=cand.backend.pricing_provider,
                model=cand.model,
                local=cand.backend.local,
            )
            cost = price.cost(usage.prompt_tokens, usage.completion_tokens)
        entry = LedgerEntry(
            endpoint=endpoint,
            requested_model=plan.requested,
            alias=plan.alias,
            backend=cand.backend.name if cand else "",
            model=cand.model if cand else "",
            status=status,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            usage_exact=usage.exact,
            cost_usd=cost,
            latency_ms=round(latency_ms, 3),
            ttft_ms=None if ttft_ms is None else round(ttft_ms, 3),
            attempts=attempts,
            tags=",".join(sorted(hints.tags)),
            error=error,
        )
        self.ledger.record(entry)
        self.metrics.observe(entry)
        return entry

    # -- public API -----------------------------------------------------------------------------

    async def chat(self, request: ChatRequest) -> tuple[ChatResult, LedgerEntry]:
        plan = self.plan(
            request.model,
            request.hints,
            prompt_tokens=estimate_tokens(request.prompt_text()),
            max_tokens=request.max_tokens,
        )
        started = time.perf_counter()
        try:
            result, cand, attempts, _ = await self._try_candidates(
                plan, lambda c: c.backend.chat(request.with_model(c.model))
            )
        except AllBackendsFailed as exc:
            last = plan.candidates[-1]
            self._entry(
                "chat",
                plan,
                last,
                request.hints,
                usage=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                attempts=len(exc.attempts),
                status="error",
                error=exc.message[:500],
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        result.backend = cand.backend.name
        result.model = result.model or cand.model
        entry = self._entry(
            "chat",
            plan,
            cand,
            request.hints,
            usage=result.usage,
            latency_ms=latency_ms,
            attempts=attempts,
        )
        return result, entry

    async def chat_stream(
        self,
        request: ChatRequest,
        *,
        on_route: Callable[[Candidate], None] | None = None,
        on_done: Callable[[LedgerEntry], None] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        plan = self.plan(
            request.model,
            request.hints,
            prompt_tokens=estimate_tokens(request.prompt_text()),
            max_tokens=request.max_tokens,
        )
        started = time.perf_counter()
        failures: list[str] = []
        attempts = 0

        for cand in plan.candidates:
            for retry in range(self.policy.retries + 1):
                attempts += 1
                stream = cand.backend.chat_stream(request.with_model(cand.model))
                try:
                    first = await anext(stream)
                except StopAsyncIteration:
                    first = None
                except BackendError as exc:
                    failures.append(f"{cand.qualified}: {exc.message}")
                    if exc.retriable and retry < self.policy.retries:
                        await self._sleep(self._backoff_seconds(retry))
                        continue
                    break
                except Exception as exc:
                    failures.append(f"{cand.qualified}: {type(exc).__name__}: {exc}")
                    log.exception("backend %s raised while opening stream", cand.qualified)
                    break

                # Committed: the first chunk arrived, fallback is no longer possible.
                ttft_ms = (time.perf_counter() - started) * 1000
                if on_route:
                    on_route(cand)
                usage: Usage | None = None
                parts: list[str] = []
                status, error = "ok", None
                try:
                    if first is not None:
                        if first.delta:
                            parts.append(first.delta)
                        usage = first.usage or usage
                        yield first
                    async for chunk in stream:
                        if chunk.delta or chunk.reasoning:
                            parts.append((chunk.delta or "") + (chunk.reasoning or ""))
                        if chunk.usage:
                            usage = chunk.usage
                        yield chunk
                except BackendError as exc:
                    status, error = "error", exc.message[:500]
                    raise
                finally:
                    if usage is None:
                        usage = Usage(
                            prompt_tokens=estimate_tokens(request.prompt_text()),
                            completion_tokens=estimate_tokens("".join(parts)),
                            exact=False,
                        )
                    entry = self._entry(
                        "chat_stream",
                        plan,
                        cand,
                        request.hints,
                        usage=usage,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        ttft_ms=ttft_ms,
                        attempts=attempts,
                        status=status,
                        error=error,
                    )
                    if on_done:
                        on_done(entry)
                return

        exc = AllBackendsFailed(failures)
        self._entry(
            "chat_stream",
            plan,
            plan.candidates[-1],
            request.hints,
            usage=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=attempts,
            status="error",
            error=exc.message[:500],
        )
        raise exc

    async def embed(self, request: EmbeddingRequest) -> tuple[EmbeddingResult, LedgerEntry]:
        plan = self.plan(
            request.model,
            request.hints,
            prompt_tokens=estimate_tokens(request.prompt_text()),
            max_tokens=0,
        )
        started = time.perf_counter()
        try:
            result, cand, attempts, _ = await self._try_candidates(
                plan, lambda c: c.backend.embed(request.with_model(c.model))
            )
        except AllBackendsFailed as exc:
            self._entry(
                "embeddings",
                plan,
                plan.candidates[-1],
                request.hints,
                usage=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                attempts=len(exc.attempts),
                status="error",
                error=exc.message[:500],
            )
            raise
        result.backend = cand.backend.name
        result.model = result.model or cand.model
        entry = self._entry(
            "embeddings",
            plan,
            cand,
            request.hints,
            usage=result.usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=attempts,
        )
        return result, entry

    # -- introspection ----------------------------------------------------------------------------

    def model_cards(self) -> list[dict[str, Any]]:
        """Everything a client may put in `model`, in OpenAI /v1/models shape plus extras."""
        created = int(time.time())
        cards: list[dict[str, Any]] = []
        for alias, chain in self.config.aliases.items():
            active = [e for e in chain if e.partition("/")[0] in self.backends]
            cards.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": created,
                    "owned_by": "openinfer",
                    "openinfer": {"kind": "alias", "chain": active, "configured": chain},
                }
            )
        for name, backend in self.backends.items():
            for model in backend.models:
                cards.append(_model_card(name, backend, model, "chat", created))
            for model in backend.embedding_models:
                cards.append(_model_card(name, backend, model, "embedding", created))
        return cards

    async def health(self) -> dict[str, HealthStatus]:
        async def probe(name: str, backend: Backend) -> tuple[str, HealthStatus]:
            try:
                return name, await asyncio.wait_for(backend.health(), timeout=3.0)
            except TimeoutError:
                return name, HealthStatus(ok=False, detail="health probe timed out")
            except Exception as exc:  # noqa: BLE001
                return name, HealthStatus(ok=False, detail=f"{type(exc).__name__}: {exc}")

        results = await asyncio.gather(*(probe(n, b) for n, b in self.backends.items()))
        return dict(results)

    async def aclose(self) -> None:
        await asyncio.gather(*(b.aclose() for b in self.backends.values()), return_exceptions=True)


def _model_card(
    backend_name: str, backend: Backend, model: str, kind: str, created: int
) -> dict[str, Any]:
    return {
        "id": f"{backend_name}/{model}",
        "object": "model",
        "created": created,
        "owned_by": backend_name,
        "openinfer": {
            "kind": kind,
            "backend": backend_name,
            "backend_type": backend.kind,
            "local": backend.local,
            "model": model,
        },
    }


def _min_opt(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)

"""Router policy tests: aliases, privacy, fallback, retries, cost and latency ceilings."""

from __future__ import annotations

import pytest

from openinfer.config import ModelPrice
from openinfer.errors import (
    AllBackendsFailed,
    BackendError,
    BackendUnavailable,
    ModelNotFound,
    NoCandidates,
    RateLimited,
)
from openinfer.ledger.base import LedgerEntry
from openinfer.pricing import Pricing
from openinfer.types import ChatRequest, EmbeddingRequest, RoutingHints
from tests.conftest import make_config, make_router
from tests.fakes import FakeBackend

POLICY_YAML = """
ledger: {url: "sqlite:///:memory:"}
backends:
  cloud: {type: echo}
  local: {type: echo}
  echo: {type: echo}
aliases:
  fast: [cloud/cloud-model, local/local-model, echo/echo]
  private: [local/local-model, echo/echo]
  cloudonly: [cloud/cloud-model]
  embed: [local/local-embed]
policy:
  default_alias: fast
  retries: 2
  backoff_base_ms: 100
  backoff_max_ms: 1000
  default_max_tokens: 1000
"""


def req(model: str = "fast", tags: set[str] | None = None, **hint_kwargs) -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=[{"role": "user", "content": "hello " * 40}],
        hints=RoutingHints(tags=frozenset(tags or set()), **hint_kwargs),
    )


def build(cloud_script=None, local_script=None, **cloud_kwargs):
    cloud = FakeBackend(
        "cloud",
        local=False,
        models=["cloud-model"],
        script=cloud_script,
        pricing_provider="groq",
        expected_latency_ms=cloud_kwargs.pop("expected_latency_ms", 500),
        **cloud_kwargs,
    )
    local = FakeBackend(
        "local",
        local=True,
        models=["local-model"],
        embedding_models=["local-embed"],
        script=local_script,
        expected_latency_ms=2000,
    )
    echo = FakeBackend("echo", local=True, models=["echo"])
    return {"cloud": cloud, "local": local, "echo": echo}


# -- resolution ------------------------------------------------------------------------------------


def test_alias_resolves_in_order_and_skips_inactive_backends():
    backends = build()
    del backends["cloud"]  # e.g. no API key -> not instantiated
    router, _, _ = make_router(make_config(POLICY_YAML), backends)
    alias, cands = router.resolve("fast")
    assert alias == "fast"
    assert [c.qualified for c in cands] == ["local/local-model", "echo/echo"]


def test_qualified_and_bare_model_names():
    router, _, _ = make_router(make_config(POLICY_YAML), build())
    assert [c.qualified for c in router.resolve("local/local-model")[1]] == ["local/local-model"]
    assert [c.qualified for c in router.resolve("cloud-model")[1]] == ["cloud/cloud-model"]
    # Unknown backend prefix is treated as a bare name and fails cleanly.
    with pytest.raises(ModelNotFound):
        router.resolve("nope/whatever")
    with pytest.raises(ModelNotFound):
        router.resolve("gpt-9")


def test_empty_model_uses_default_alias():
    router, _, _ = make_router(make_config(POLICY_YAML), build())
    alias, _ = router.resolve("")
    assert alias == "fast"


# -- privacy ---------------------------------------------------------------------------------------


async def test_private_tag_removes_cloud_candidates():
    backends = build()
    router, _, _ = make_router(make_config(POLICY_YAML), backends)
    result, entry = await router.chat(req("fast", tags={"private"}))
    assert result.backend == "local"
    assert backends["cloud"].calls == []  # never contacted
    assert entry.tags == "private"


async def test_private_alias_name_is_treated_as_private_tag():
    backends = build()
    router, _, _ = make_router(make_config(POLICY_YAML), backends)
    plan = router.plan("private", RoutingHints())
    assert all(c.backend.local for c in plan.candidates)


async def test_private_with_only_cloud_candidates_fails_closed():
    backends = build()
    router, _, _ = make_router(make_config(POLICY_YAML), backends)
    with pytest.raises(NoCandidates) as exc:
        await router.chat(req("cloudonly", tags={"private"}))
    assert exc.value.status_code == 422
    assert "Refusing to send data off-machine" in exc.value.message
    assert backends["cloud"].calls == []


# -- fallback and retries --------------------------------------------------------------------------


async def test_unavailable_backend_falls_through_without_retry():
    backends = build(cloud_script=[BackendUnavailable("down", backend="cloud")])
    router, sleeper, ledger = make_router(make_config(POLICY_YAML), backends)
    result, entry = await router.chat(req())
    assert result.backend == "local"
    assert entry.attempts == 2
    assert sleeper.delays == []  # non-retriable: no backoff wait
    assert len(backends["cloud"].calls) == 1
    assert ledger.recent(1)[0]["backend"] == "local"


async def test_rate_limit_retries_same_backend_with_growing_backoff():
    backends = build(
        cloud_script=[
            RateLimited("slow down", backend="cloud"),
            RateLimited("again", backend="cloud"),
        ]
    )
    router, sleeper, _ = make_router(make_config(POLICY_YAML), backends)
    result, entry = await router.chat(req())
    assert result.backend == "cloud"
    assert entry.attempts == 3
    assert len(sleeper.delays) == 2
    assert 0.1 <= sleeper.delays[0] < 0.2  # base 100ms + jitter up to 50ms
    assert 0.2 <= sleeper.delays[1] < 0.3  # doubled
    assert len(backends["local"].calls) == 0


async def test_retries_exhausted_then_next_candidate():
    backends = build(cloud_script=[RateLimited("x", backend="cloud")] * 3)
    router, sleeper, _ = make_router(make_config(POLICY_YAML), backends)
    result, entry = await router.chat(req())
    assert result.backend == "local"
    assert entry.attempts == 4  # 3 on cloud (1 + 2 retries) then local
    assert len(sleeper.delays) == 2


async def test_backoff_is_capped():
    router, _, _ = make_router(make_config(POLICY_YAML), build())
    assert router._backoff_seconds(20) <= 1.0 + 0.05


async def test_all_backends_fail_records_error_entry():
    backends = build(
        cloud_script=[BackendError("bad", backend="cloud", retriable=False)],
        local_script=[BackendUnavailable("gone", backend="local")],
    )
    backends["echo"].script = [BackendUnavailable("gone too", backend="echo")]
    router, _, ledger = make_router(make_config(POLICY_YAML), backends)
    with pytest.raises(AllBackendsFailed) as exc:
        await router.chat(req())
    assert exc.value.status_code == 502
    assert "cloud/cloud-model: bad" in exc.value.message
    row = ledger.recent(1)[0]
    assert row["status"] == "error"
    assert row["attempts"] == 3


async def test_unexpected_exception_from_adapter_moves_on():
    backends = build(cloud_script=[RuntimeError("adapter bug")])
    router, _, _ = make_router(make_config(POLICY_YAML), backends)
    result, entry = await router.chat(req())
    assert result.backend == "local"
    assert entry.attempts == 2


# -- cost ceiling ----------------------------------------------------------------------------------


def pricing_with_cloud(price_in: float, price_out: float) -> Pricing:
    p = Pricing({"groq": {"cloud-model": ModelPrice(input=price_in, output=price_out)}})
    return p


async def test_cost_ceiling_drops_expensive_cloud_but_keeps_local():
    backends = build()
    # $10 per 1M output tokens; default_max_tokens 1000 -> about $0.01 per request
    router, _, _ = make_router(
        make_config(POLICY_YAML), backends, pricing=pricing_with_cloud(1.0, 10.0)
    )
    plan = router.plan("fast", RoutingHints(max_cost_usd=0.001), prompt_tokens=100)
    assert [c.qualified for c in plan.candidates] == ["local/local-model", "echo/echo"]
    assert plan.rejected[0][0] == "cloud/cloud-model"
    assert "cost" in plan.rejected[0][1]


async def test_cost_ceiling_keeps_cheap_cloud():
    router, _, _ = make_router(
        make_config(POLICY_YAML), build(), pricing=pricing_with_cloud(0.05, 0.08)
    )
    plan = router.plan("fast", RoutingHints(max_cost_usd=0.001), prompt_tokens=100)
    assert plan.candidates[0].qualified == "cloud/cloud-model"


async def test_global_policy_cost_ceiling_applies_without_hint():
    yaml = POLICY_YAML + "  max_cost_usd: 0.0001\n"
    router, _, _ = make_router(make_config(yaml), build(), pricing=pricing_with_cloud(1.0, 10.0))
    result, _ = await router.chat(req())
    assert result.backend == "local"


async def test_cost_ceiling_with_only_expensive_candidates_fails():
    router, _, _ = make_router(
        make_config(POLICY_YAML), build(), pricing=pricing_with_cloud(1.0, 10.0)
    )
    with pytest.raises(NoCandidates) as exc:
        router.plan("cloudonly", RoutingHints(max_cost_usd=0.000001), prompt_tokens=1000)
    assert "cost" in exc.value.message


async def test_ledger_entry_has_cost_from_pricing():
    router, _, ledger = make_router(
        make_config(POLICY_YAML), build(), pricing=pricing_with_cloud(1.0, 2.0)
    )
    _, entry = await router.chat(req())
    # FakeBackend reports 10 prompt + 20 completion tokens
    assert entry.cost_usd == pytest.approx((10 * 1.0 + 20 * 2.0) / 1_000_000)
    assert entry.backend == "cloud"
    assert ledger.stats()["totals"]["cost_usd"] == pytest.approx(entry.cost_usd, abs=1e-6)


# -- latency ceiling -------------------------------------------------------------------------------


async def test_latency_ceiling_uses_configured_expectation():
    router, _, _ = make_router(make_config(POLICY_YAML), build())
    # cloud expects 500ms, local 2000ms, echo unknown (kept)
    plan = router.plan("fast", RoutingHints(max_latency_ms=1000))
    assert [c.qualified for c in plan.candidates] == ["cloud/cloud-model", "echo/echo"]


async def test_latency_ceiling_prefers_measured_p50_once_enough_samples():
    router, _, ledger = make_router(make_config(POLICY_YAML), build())
    for _ in range(5):
        ledger.record(
            LedgerEntry(
                endpoint="chat",
                requested_model="fast",
                backend="cloud",
                model="cloud-model",
                latency_ms=5000,
            )
        )
    plan = router.plan("fast", RoutingHints(max_latency_ms=1000))
    assert "cloud/cloud-model" not in [c.qualified for c in plan.candidates]
    assert any("latency" in reason for _, reason in plan.rejected)


# -- streaming -------------------------------------------------------------------------------------


async def test_stream_falls_back_before_first_chunk_and_records_ttft():
    backends = build(cloud_script=[BackendUnavailable("down", backend="cloud")])
    router, _, ledger = make_router(make_config(POLICY_YAML), backends)
    routed: list[str] = []
    chunks = [
        c
        async for c in router.chat_stream(
            req(), on_route=lambda cand: routed.append(cand.backend.name)
        )
    ]
    assert routed == ["local"]
    assert "".join(c.delta or "" for c in chunks) == "w0 w1 w2 "
    row = ledger.recent(1)[0]
    assert row["endpoint"] == "chat_stream"
    assert row["backend"] == "local"
    assert row["attempts"] == 2
    assert row["ttft_ms"] is not None
    assert row["completion_tokens"] == 3  # from the final chunk's usage


async def test_stream_all_fail_raises():
    backends = build(
        cloud_script=[BackendUnavailable("a", backend="cloud")],
        local_script=[BackendUnavailable("b", backend="local")],
    )
    backends["echo"].script = [BackendUnavailable("c", backend="echo")]
    router, _, _ = make_router(make_config(POLICY_YAML), backends)
    with pytest.raises(AllBackendsFailed):
        async for _ in router.chat_stream(req()):
            pass


# -- embeddings ------------------------------------------------------------------------------------


async def test_embed_routes_through_alias():
    router, _, _ = make_router(make_config(POLICY_YAML), build())
    result, entry = await router.embed(
        EmbeddingRequest(model="embed", inputs=["a", "b"], hints=RoutingHints())
    )
    assert result.backend == "local"
    assert len(result.vectors) == 2
    assert entry.endpoint == "embeddings"


# -- model cards -----------------------------------------------------------------------------------


def test_model_cards_list_aliases_then_concrete_models():
    router, _, _ = make_router(make_config(POLICY_YAML), build())
    ids = [c["id"] for c in router.model_cards()]
    assert ids[:4] == ["fast", "private", "cloudonly", "embed"]
    assert "cloud/cloud-model" in ids
    assert "local/local-embed" in ids
    card = next(c for c in router.model_cards() if c["id"] == "cloud/cloud-model")
    assert card["openinfer"]["local"] is False

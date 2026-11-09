from __future__ import annotations

from openinfer.backends import build_backends
from openinfer.bench import markdown_table, run_bench
from openinfer.errors import BackendUnavailable
from openinfer.types import HealthStatus
from tests.conftest import make_config
from tests.fakes import FakeBackend


async def test_bench_echo_produces_markdown_table():
    cfg = make_config()
    backends = build_backends(cfg)
    results = await run_bench(cfg, backends, runs=2, max_tokens=16, prompt="bench me")
    assert len(results) == 1
    r = results[0]
    assert r.backend == "echo" and r.status == "ok" and r.runs == 2
    assert r.tokens_per_s is not None and r.tokens_per_s > 0
    assert r.ttft_ms_p50 is not None and r.usd_per_1m_output == 0.0
    table = markdown_table(results)
    lines = table.splitlines()
    assert lines[0].startswith("| Backend | Model |")
    assert lines[1].startswith("|---")
    assert "| echo | echo | local |" in lines[2]


async def test_bench_marks_unreachable_backends():
    cfg = make_config()
    down = FakeBackend("down", models=["m"])

    async def unhealthy() -> HealthStatus:
        return HealthStatus(ok=False, detail="connection refused")

    down.health = unhealthy  # type: ignore[method-assign]
    results = await run_bench(cfg, {"down": down}, runs=1)
    assert results[0].status == "unreachable"
    assert "connection refused" in markdown_table(results)


async def test_bench_reports_errors_mid_run():
    cfg = make_config()
    flaky = FakeBackend("flaky", models=["m"], script=[None, BackendUnavailable("gone")])
    results = await run_bench(cfg, {"flaky": flaky}, runs=3, skip_unhealthy=False)
    assert results[0].status == "error" and results[0].runs == 1


async def test_bench_only_filter():
    cfg = make_config()
    a, b = FakeBackend("a", models=["m"]), FakeBackend("b", models=["m"])
    results = await run_bench(cfg, {"a": a, "b": b}, only=["b"], runs=1)
    assert [r.backend for r in results] == ["b"]

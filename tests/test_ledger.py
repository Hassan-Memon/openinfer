from __future__ import annotations

import time

import pytest

from openinfer.ledger import open_ledger
from openinfer.ledger.base import LedgerEntry, percentile
from openinfer.ledger.sqlite import SQLiteLedger, sqlite_path


def entry(**kwargs) -> LedgerEntry:
    base = dict(
        endpoint="chat",
        requested_model="fast",
        backend="echo",
        model="echo",
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.001,
        latency_ms=100.0,
    )
    base.update(kwargs)
    return LedgerEntry(**base)


def test_sqlite_path_parsing():
    assert sqlite_path("sqlite:///./x.db") == "./x.db"
    assert sqlite_path("sqlite:////data/x.db") == "/data/x.db"
    assert sqlite_path("sqlite:///:memory:") == ":memory:"
    assert sqlite_path("sqlite:///") == ":memory:"
    with pytest.raises(ValueError, match="not a sqlite url"):
        sqlite_path("postgresql://x")


def test_open_ledger_dispatch(tmp_path):
    led = open_ledger(f"sqlite:///{(tmp_path / 'l.db').as_posix()}")
    assert isinstance(led, SQLiteLedger)
    led.close()
    with pytest.raises(ValueError, match="unsupported"):
        open_ledger("mysql://nope")


def test_postgres_requires_extra():
    try:
        import psycopg  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="postgres"):
            open_ledger("postgresql://u:p@localhost/db")
    else:
        pytest.skip("psycopg installed; would attempt a real connection")


def test_record_recent_and_stats(tmp_path):
    led = SQLiteLedger(f"sqlite:///{(tmp_path / 'ledger.db').as_posix()}")
    led.record(entry(latency_ms=100))
    led.record(entry(latency_ms=300, ttft_ms=40))
    led.record(entry(status="error", error="boom", latency_ms=50, cost_usd=0))
    led.record(entry(backend="cloud", model="m", cost_usd=0.01, ts=time.time() - 48 * 3600))

    recent = led.recent(2)
    assert len(recent) == 2
    assert recent[0]["status"] == "error"  # newest first; the cloud row is 48h old
    stats = led.stats(hours=24)
    assert stats["totals"]["requests"] == 3
    assert stats["totals"]["errors"] == 1
    assert stats["totals"]["prompt_tokens"] == 30
    assert stats["totals"]["cost_usd"] == pytest.approx(0.002)
    assert stats["totals"]["p50_latency_ms"] == 200.0
    assert stats["totals"]["p50_ttft_ms"] == 40.0
    assert stats["by_backend"][0]["backend"] == "echo"
    assert stats["by_backend"][0]["requests"] == 3
    all_time = led.stats(hours=24 * 30)
    assert all_time["totals"]["requests"] == 4
    led.close()


def test_recent_orders_by_timestamp():
    led = SQLiteLedger()
    led.record(entry(backend="old", ts=1.0))
    led.record(entry(backend="new", ts=2.0))
    assert [r["backend"] for r in led.recent(5)] == ["new", "old"]


def test_latency_p50_needs_min_samples():
    led = SQLiteLedger()
    for latency in (100, 200, 300, 400):
        led.record(entry(latency_ms=latency))
    assert led.latency_p50("echo", "echo", min_samples=5) is None
    led.record(entry(latency_ms=500))
    assert led.latency_p50("echo", "echo", min_samples=5) == 300.0
    led.record(entry(status="error", latency_ms=99999))  # errors ignored
    assert led.latency_p50("echo", "echo", min_samples=5) == 300.0
    assert led.latency_p50("other", "echo") is None


def test_percentile():
    assert percentile([], 0.5) is None
    assert percentile([10.0], 0.95) == 10.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5

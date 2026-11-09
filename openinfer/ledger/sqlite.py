"""SQLite ledger using only the standard library.

One connection, one lock. Writes are a few per request, so a mutex is plenty; WAL mode lets
readers (the /v1/ledger endpoint) run while writes happen.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from openinfer.ledger.base import Ledger, LedgerEntry, percentile

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    endpoint TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    alias TEXT,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    usage_exact INTEGER NOT NULL DEFAULT 1,
    cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    ttft_ms REAL,
    attempts INTEGER NOT NULL DEFAULT 1,
    tags TEXT NOT NULL DEFAULT '',
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts);
CREATE INDEX IF NOT EXISTS idx_requests_backend_model ON requests (backend, model, ts);
"""

COLUMNS = (
    "ts",
    "endpoint",
    "requested_model",
    "alias",
    "backend",
    "model",
    "status",
    "prompt_tokens",
    "completion_tokens",
    "usage_exact",
    "cost_usd",
    "latency_ms",
    "ttft_ms",
    "attempts",
    "tags",
    "error",
)


def sqlite_path(url: str) -> str:
    """sqlite:///./x.db -> ./x.db ; sqlite:////data/x.db -> /data/x.db ; sqlite:///:memory:"""
    if not url.startswith("sqlite:///"):
        raise ValueError(f"not a sqlite url: {url!r}")
    return url[len("sqlite:///") :] or ":memory:"


class SQLiteLedger(Ledger):
    placeholder = "?"

    def __init__(self, url: str = "sqlite:///:memory:") -> None:
        self.path = sqlite_path(url)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)

    # -- writes ---------------------------------------------------------------------------

    def record(self, entry: LedgerEntry) -> None:
        values = (
            entry.ts,
            entry.endpoint,
            entry.requested_model,
            entry.alias,
            entry.backend,
            entry.model,
            entry.status,
            entry.prompt_tokens,
            entry.completion_tokens,
            1 if entry.usage_exact else 0,
            entry.cost_usd,
            entry.latency_ms,
            entry.ttft_ms,
            entry.attempts,
            entry.tags,
            entry.error,
        )
        marks = ", ".join([self.placeholder] * len(COLUMNS))
        sql = f"INSERT INTO requests ({', '.join(COLUMNS)}) VALUES ({marks})"
        with self._lock:
            self._execute(sql, values)
            self._commit()

    # -- reads -----------------------------------------------------------------------------

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM requests ORDER BY ts DESC LIMIT {self.placeholder}"
        with self._lock:
            rows = self._fetchall(sql, (int(limit),))
        return [self._row_to_dict(r) for r in rows]

    def latency_p50(self, backend: str, model: str, *, min_samples: int = 5) -> float | None:
        p = self.placeholder
        sql = (
            "SELECT latency_ms FROM requests WHERE backend = "
            f"{p} AND model = {p} AND status = 'ok' ORDER BY ts DESC LIMIT 200"
        )
        with self._lock:
            rows = self._fetchall(sql, (backend, model))
        values = [float(r[0]) for r in rows]
        if len(values) < min_samples:
            return None
        return percentile(values, 0.5)

    def stats(self, hours: float = 24.0) -> dict[str, Any]:
        since = time.time() - hours * 3600
        p = self.placeholder
        with self._lock:
            totals = self._fetchone(
                "SELECT COUNT(*) AS requests, "
                "SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS errors, "
                "COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
                "COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
                "COALESCE(SUM(cost_usd), 0) AS cost_usd "
                f"FROM requests WHERE ts >= {p}",
                (since,),
            )
            latencies = [
                float(r[0])
                for r in self._fetchall(
                    f"SELECT latency_ms FROM requests WHERE ts >= {p} AND status = 'ok' "
                    "ORDER BY ts DESC LIMIT 10000",
                    (since,),
                )
            ]
            ttfts = [
                float(r[0])
                for r in self._fetchall(
                    f"SELECT ttft_ms FROM requests WHERE ts >= {p} AND status = 'ok' "
                    "AND ttft_ms IS NOT NULL ORDER BY ts DESC LIMIT 10000",
                    (since,),
                )
            ]
            by_backend = self._fetchall(
                "SELECT backend, model, COUNT(*) AS requests, "
                "SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS errors, "
                "COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
                "COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
                "COALESCE(SUM(cost_usd), 0) AS cost_usd, "
                "AVG(latency_ms) AS avg_latency_ms "
                f"FROM requests WHERE ts >= {p} GROUP BY backend, model ORDER BY requests DESC",
                (since,),
            )
        totals_d = self._row_to_dict(totals) if totals else {}
        return {
            "window_hours": hours,
            "totals": {
                "requests": int(totals_d.get("requests") or 0),
                "errors": int(totals_d.get("errors") or 0),
                "prompt_tokens": int(totals_d.get("prompt_tokens") or 0),
                "completion_tokens": int(totals_d.get("completion_tokens") or 0),
                "cost_usd": round(float(totals_d.get("cost_usd") or 0.0), 6),
                "p50_latency_ms": _round(percentile(latencies, 0.5)),
                "p95_latency_ms": _round(percentile(latencies, 0.95)),
                "p50_ttft_ms": _round(percentile(ttfts, 0.5)),
            },
            "by_backend": [
                {
                    "backend": r["backend"],
                    "model": r["model"],
                    "requests": int(r["requests"]),
                    "errors": int(r["errors"] or 0),
                    "prompt_tokens": int(r["prompt_tokens"] or 0),
                    "completion_tokens": int(r["completion_tokens"] or 0),
                    "cost_usd": round(float(r["cost_usd"] or 0.0), 6),
                    "avg_latency_ms": _round(r["avg_latency_ms"]),
                }
                for r in (self._row_to_dict(x) for x in by_backend)
            ],
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- driver specific bits (overridden by PostgresLedger) ---------------------------------

    def _execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self._conn.execute(sql, params)

    def _commit(self) -> None:
        self._conn.commit()

    def _fetchall(self, sql: str, params: tuple[Any, ...]) -> list[Any]:
        return list(self._conn.execute(sql, params).fetchall())

    def _fetchone(self, sql: str, params: tuple[Any, ...]) -> Any:
        return self._conn.execute(sql, params).fetchone()

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        return dict(row)


def _round(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(float(value), digits)

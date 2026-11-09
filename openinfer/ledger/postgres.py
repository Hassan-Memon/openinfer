"""Postgres ledger (optional extra `postgres`, uses psycopg 3).

Shares all SQL with the SQLite ledger; only the connection, placeholder style and the
autoincrement column differ. Neon and Supabase free tiers both work.
"""

from __future__ import annotations

import threading
from typing import Any

from openinfer.ledger.sqlite import SCHEMA, SQLiteLedger


class PostgresLedger(SQLiteLedger):
    placeholder = "%s"

    def __init__(self, url: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "Postgres ledger requires the 'postgres' extra: uv sync --extra postgres"
            ) from exc
        self.path = url
        self._lock = threading.Lock()
        self._conn = psycopg.connect(url, row_factory=dict_row, autocommit=False)
        schema = SCHEMA.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
        schema = schema.replace("REAL", "DOUBLE PRECISION")
        with self._lock, self._conn.cursor() as cur:
            cur.execute(schema)
            self._conn.commit()

    def _execute(self, sql: str, params: tuple[Any, ...]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)

    def _fetchall(self, sql: str, params: tuple[Any, ...]) -> list[Any]:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        # Positional access (row[0]) is used for single column selects.
        return [_PositionalDict(r) for r in rows]

    def _fetchone(self, sql: str, params: tuple[Any, ...]) -> Any:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return _PositionalDict(row) if row else None


class _PositionalDict(dict):
    """dict that also answers row[0] like sqlite3.Row does."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)

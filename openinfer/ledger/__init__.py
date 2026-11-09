"""Cost and latency ledger.

Every routed request writes one row: who served it, how many tokens, what it cost, how long
it took. No prompt or completion text is stored. SQLite is the default (a single file, no
service to run); Postgres is a drop in when several replicas must share one ledger.
"""

from __future__ import annotations

from openinfer.ledger.base import Ledger, LedgerEntry
from openinfer.ledger.sqlite import SQLiteLedger


def open_ledger(url: str) -> Ledger:
    """Open a ledger from a URL: sqlite:///path.db, sqlite:///:memory:, postgresql://..."""
    if url.startswith("sqlite://"):
        return SQLiteLedger(url)
    if url.startswith(("postgresql://", "postgres://")):
        from openinfer.ledger.postgres import PostgresLedger

        return PostgresLedger(url)
    raise ValueError(f"unsupported ledger url: {url!r}")


__all__ = ["Ledger", "LedgerEntry", "SQLiteLedger", "open_ledger"]

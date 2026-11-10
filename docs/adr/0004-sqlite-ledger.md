# ADR-0004: SQLite as the default cost ledger, Postgres optional

- Status: accepted
- Date: 2026-11-09
- Deciders: Hassan Memon

## Context and problem statement

Every routed request should leave a record: backend, model, tokens, estimated cost, latency,
time to first token, attempts, status. The router also reads back recent latency to enforce
`max_latency_ms`. Where does this live?

## Decision drivers

- `uvx --from git+https://github.com/Hassan-Memon/openinfer openinfer serve` must work with no services running.
- Writes are small and frequent (one row per request); reads are aggregates over hours or days.
- Several replicas behind a load balancer should be able to share one ledger eventually.
- No prompt or completion text is stored, so the data is low sensitivity but still operational.
- Free hosting: Neon and Supabase give free Postgres; Hugging Face and Render give ephemeral disks.

## Considered options

1. **In-memory only**: trivial, lost on restart, useless for cost review.
2. **SQLite file** via the standard library `sqlite3`: zero dependencies, one file, WAL mode for
   concurrent readers, good enough for thousands of requests per minute from one process.
3. **Postgres** via psycopg: shared, durable, more moving parts.
4. **An ORM (SQLAlchemy / SQLModel)** over either: adds a dependency and migration tooling for one
   table.
5. **Prometheus only**: great for rates and histograms, cannot answer "what did the request at
   14:02 cost" or do per-model totals over arbitrary windows without a long-term store.

## Decision

Options 2 and 3 behind one interface (`openinfer/ledger/base.py`), chosen by URL:
`sqlite:///./openinfer.db` by default, `postgresql://...` with the `postgres` extra. The SQL is
shared; the Postgres class overrides only the connection, placeholder style and the
autoincrement type (`openinfer/ledger/postgres.py`). `sqlite:///:memory:` is used by tests and
by the CLI commands that need a router but no persistence. Prometheus metrics are emitted in
parallel (`openinfer/metrics.py`) for dashboards, with the ledger as the source of record.

Percentiles are computed in Python over the last N rows rather than with database-specific
functions, so both engines behave identically.

## Consequences

- Positive: no service to run, no migration tool; the schema is created on first open.
- Positive: `/v1/ledger` and `openinfer bench` have a single, testable data source.
- Negative: SQLite is single-writer. Several replicas each get their own file unless pointed at
  Postgres; the compose file mounts a volume so at least one container keeps history.
- Negative: no retention policy yet; a `DELETE WHERE ts < ?` job is listed in the exercises.
- Neutral: the ledger stores estimates when a backend reports no usage (`usage_exact = 0`), so
  cost totals should be read with that column in mind.

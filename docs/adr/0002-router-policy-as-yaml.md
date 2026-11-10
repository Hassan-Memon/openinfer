# ADR-0002: Routing policy as a declarative YAML file

- Status: accepted
- Date: 2026-11-09
- Deciders: Hassan Memon

## Context and problem statement

The router has to answer "which backend and model should serve this request" using the alias
requested, request tags, cost and latency limits, and which backends are currently available.
That logic could live in Python code (a plugin or callback), in a database table with an admin
UI, or in a configuration file that ships with the deployment.

## Decision drivers

- Operators, not developers, change routing. They should not need to redeploy code to add a
  backend or reorder a fallback chain.
- The policy must be reviewable in a pull request and reproducible across environments.
- Secrets must stay out of the file so it can be committed.
- Tests need to construct policies in a few lines.
- The MVP must work with zero configuration.

## Considered options

1. **Python callback** (`def route(request) -> list[Candidate]`). Maximum power, zero
   reviewability for non-developers, every change is a deploy.
2. **Database-backed rules** with an API. Dynamic, but adds a schema, migrations and an admin
   surface before the first user exists.
3. **Declarative YAML** validated by pydantic models, with environment variable interpolation
   for secrets and hosts.

## Decision

Option 3. `openinfer.yaml` declares backends, alias chains (ordered lists of `backend/model`)
and a `policy` block (private tag, ceilings, retries, backoff). `${VAR:-default}` references are
expanded before parsing (`openinfer/config.py`), with automatic quoting of values that YAML would
misread. A copy ships inside the package as the default, so `uvx --from git+https://github.com/Hassan-Memon/openinfer openinfer serve` works without
a file; a test asserts the repo example and the packaged default stay identical.

The router (`openinfer/router/engine.py`) is deliberately simple: resolve, filter, try in
order. Anything that needs runtime data (measured latency) reads it from the ledger rather than
from configuration.

## Consequences

- Positive: a policy change is a diff in a PR; CI validates the schema.
- Positive: backends with missing keys are silently inactive, so one file serves laptop, CI and
  production.
- Negative: no per-tenant or time-of-day rules without restarting; a reload endpoint is the
  natural next step and is listed in the exercises.
- Negative: ordering is static. A backend that is slow today still gets tried first unless a
  latency ceiling is set; adaptive ordering from ledger p50 is a planned extension that fits the
  model (the data is already there).
- Neutral: pydantic errors on a bad file are verbose but precise; `openinfer config` prints the
  effective result with secrets masked.

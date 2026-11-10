# ADR-0006: A deterministic echo backend as a first-class citizen

- Status: accepted
- Date: 2026-11-09
- Deciders: Hassan Memon

## Context and problem statement

Tests, CI, `docker compose up`, the Hugging Face demo and downstream projects (AgentMesh,
IntakeAgent run with `DEMO_MODE=1`) all need the API to work with no model weights, no GPU and no
API keys. Mocking HTTP in every test is brittle, and shipping a real model to CI is slow.

## Decision drivers

- CI must be fast (seconds) and offline.
- The full request path (server, router, ledger, metrics, SSE) should be exercised, not stubbed.
- Behaviour must be deterministic so assertions are exact.
- The same thing should power a public demo where cost must be zero.

## Considered options

1. **Mock transports everywhere**: tests the adapters but not the integration.
2. **A tiny real model in CI**: 400 MB download, minutes of CPU, flaky assertions on text.
3. **An `echo` backend** implementing the same `Backend` interface as real engines.

## Decision

Option 3 (`openinfer/backends/echo.py`), plus option 1 for adapter-level tests
(`tests/test_backends_http.py`). Echo mirrors the last user message (`Echo: ...`), streams word
by word, emits a tool call when `tool_choice` forces one, and returns feature-hashed unit-length
embeddings that are stable across runs. It reports estimated token usage flagged
`exact=False`, like a real backend that omits usage. An optional `delay_ms` makes streaming
visible in demos.

It is listed in the default config as the last link of every alias chain, so a fresh install
always answers. It is marked `local = True` because nothing leaves the process.

## Consequences

- Positive: 100+ tests run in a few seconds with no network; the OpenAI SDK round trip is a real
  end-to-end test.
- Positive: `uvx --from git+https://github.com/Hassan-Memon/openinfer openinfer serve` works on the first try for everyone.
- Negative: a chain that silently ends in echo can hide an outage in production. Operators
  should remove `echo/echo` from production chains or alert on
  `openinfer_requests_total{backend="echo"}`; the docs say so.
- Neutral: hashed embeddings are not semantic. They are enough to test vector plumbing and
  dimensions; anything needing meaning must use Ollama, Gemini or OpenAI embeddings.

# 02. System design trade-offs

An inference gateway is mostly waiting: on a GPU, on a remote API, on a slow CPU. The design
questions are about what happens while waiting and what happens when the wait fails.

## Scaling

The server itself is cheap. FastAPI on uvicorn handles thousands of concurrent streams per
process because each request is an idle coroutine most of the time. Run `uvicorn --workers N`
or several containers behind a load balancer and the router scales horizontally without
coordination, with one caveat: the SQLite ledger is per process. Several replicas should point
`OPENINFER_LEDGER_URL` at Postgres (Neon or Supabase free tiers work), or accept that each
replica has a partial view.

The backends are where capacity lives. llama.cpp in-process is single-stream by design: a lock
in `backends/llamacpp.py` serialises calls because one `Llama` object is not thread safe. That
is right for a laptop and wrong for 50 users. The policy makes the fix a config change: put
Ollama (which queues) or vLLM (which batches) ahead of it in the chain.

## What breaks first at 10x

1. **The local engine saturates.** Requests queue behind the lock, latency climbs, clients time
   out. Signal: `openinfer_request_latency_seconds` p95 for `backend="llamacpp"` rises while
   request count is flat. Fix: `max_latency_ms` ceilings so the router skips it once measured p50
   exceeds the budget (it reads the ledger), or move to vLLM.
2. **Cloud rate limits.** Free tiers return 429. Retries with backoff absorb bursts; sustained
   load needs more candidates in the chain or a paid tier. Signal: `openinfer_fallbacks_total`.
3. **The ledger write path.** SQLite commits per request. At a few hundred requests per second
   the mutex in `SQLiteLedger.record()` becomes visible. Fix: batch writes, or Postgres.
4. **Health probes.** `/health` probes every backend in parallel with a 3 s cap. A load balancer
   polling it every 5 s against 8 backends is 8 outbound calls per replica per 5 s. Use
   `/health?deep=false` for liveness and the deep probe for a dashboard.

## Failure modes and how each is handled

| Failure | Behaviour | Where |
|---|---|---|
| Backend down (connection refused) | Next candidate immediately, no sleep | `BackendUnavailable` in `router/engine.py` |
| 429 or 5xx or timeout | Retry same backend with exponential backoff and jitter, then next | `_backoff_seconds`, `_try_candidates` |
| Backend returns 4xx (bad model name) | Next candidate, not retried | `retriable=False` |
| Adapter raises an unexpected exception | Logged, treated as non-retriable, next candidate | `except Exception` in `_try_candidates` |
| Stream dies mid-response | Error event then `[DONE]`; ledger row has `status=error` | `routes/chat.py` |
| Every candidate fails | 502 with every attempt listed; ledger row recorded | `AllBackendsFailed` |
| Privacy filter removes everything | 422, no network call | `NoCandidates` |

The asymmetry between "retry" and "move on" is deliberate. Retrying a refused connection wastes
the user's time; moving on from a 429 wastes a backend that would have worked a second later.

## Back-pressure

There is none built in, and that is a conscious gap. uvicorn accepts connections as fast as they
arrive; a slow backend turns into many open coroutines holding memory. The standard remedies, in
order of effort: a per-backend semaphore in the adapter (vLLM and Ollama already queue
server-side), a global concurrency limit middleware that returns 429 with `Retry-After`, and a
real queue in front of the engine for batch workloads. The ledger's `attempts` and TTFT columns
tell you when you need it.

## Idempotency

Chat completions are not idempotent: the same prompt yields different text (unless
`temperature=0` and `seed` are set, which many backends honour and the request carries
through). The router retries only when it is certain nothing was generated: a 429, a 5xx, a
timeout before any byte, or a connection failure. Once the first streamed chunk arrives, the
router is committed and will not retry elsewhere, because the client has already seen partial
output and a second backend would produce a different continuation. For non-streaming, a
timeout after the backend started generating does get retried, which can double-bill tokens
on the first backend; the ledger records only what we received, so this shows up as a cost
discrepancy against the provider's invoice. Passing an idempotency key to providers that
support one is the next improvement.

## Consistency of the ledger

Writes happen after the response is complete (streaming) or just before it is sent
(non-streaming). A crash in between loses that row. This is acceptable for cost accounting
(approximate by nature, since token counts are sometimes estimated) and would not be for
billing customers. The `usage_exact` column makes the distinction visible.

## Security boundaries

Keys are environment-only. The bearer token guards `/v1/*`, not `/health` or `/metrics`, which
is the right default behind a private network and the wrong one on the open internet; the
SECURITY.md checklist covers it. Prompt text never touches disk in OpenInfer; it may well be
logged by the backend you route to, which is why `private` exists.

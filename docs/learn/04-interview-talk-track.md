# 04. Interview talk track

## The two minute version

"OpenInfer is an OpenAI-compatible inference gateway I built to solve a problem I kept hitting:
teams want frontier-model quality for most requests, a hard guarantee that sensitive requests
never leave their own hardware, and a clear view of what inference costs. Existing tools give
you one of those. OpenInfer gives all three in one small Python service.

Clients keep their OpenAI SDK and just change the base URL. Instead of a vendor model they ask
for an alias like `fast`, `smart` or `private`. A YAML policy maps each alias to an ordered
chain of backends: Groq, Gemini, OpenRouter free tiers, Ollama, vLLM, or llama.cpp running a
GGUF in-process on whatever accelerator the box has: CPU, Apple Metal, CUDA, ROCm. The router
filters that chain per request. A `private` tag removes every cloud candidate and fails closed
if nothing local is up; cost and latency ceilings prune the rest using a price table and the
measured p50 from a ledger. Then it tries candidates in order with exponential backoff for rate
limits and immediate fallback for outages.

Every request writes one ledger row: backend, tokens, cost, latency, time to first token. That
feeds `/v1/ledger`, Prometheus `/metrics` and a `bench` command that prints tokens per second
and dollars per million tokens per backend. It runs entirely on free tiers: `docker compose up`
with zero keys gives you a working API on a deterministic echo backend, and the whole test suite
(over 100 tests, including the real `openai` SDK driving the app) runs offline in seconds."

## The fifteen minute version

Walk the request path from `docs/learn/01-architecture-walkthrough.md`, then spend time on
three decisions and one honest limitation.

**Decision 1: OpenAI compatibility (ADR-0001).** Why: adoption cost is a URL. How: loose
validation, extensions in headers and an `openinfer` object, internal dataclasses as the seam.
Proof: `tests/test_openai_sdk.py` runs the real SDK through the ASGI app, including streaming
and base64 embeddings, which the SDK requests by default and which I had to implement after the
test caught it.

**Decision 2: privacy as a code invariant (ADR-0005).** `local` is a class attribute on the
adapter, not config. The filter runs before any network call. Fail closed, 422, names the
rejected candidates. Show `test_private_with_only_cloud_candidates_fails_closed`.

**Decision 3: llama.cpp first, vLLM and Ollama over HTTP (ADR-0003).** One GGUF runs on every
accelerator with one flag; `openinfer hardware` tells you the build command. vLLM wins at
concurrency and is a 20 line preset. Be clear that NPUs are detected, not used.

**Limitation to volunteer:** no back-pressure. uvicorn accepts everything; the llama.cpp lock
serialises. The ledger shows it (TTFT climbs) and the fix is a semaphore per backend or a
`max_latency_ms` ceiling that routes around the saturated engine. Volunteering this is better
than being asked.

## Numbers to quote

- 111 tests, all offline, about 7 seconds on a laptop.
- Real local model on this CPU laptop via Ollama: deepseek-r1 1.5B, 380 ms TTFT, 16 tokens/s.
- Groq llama-3.1-8b: $0.08 per million output tokens. GPT-4o: $10. A single-stream self-hosted
  g5.xlarge at 75 tokens/s: about $3.70 per million; batched with vLLM at 50 percent
  utilisation: about $0.25.
- Oracle Always Free: 4 ARM cores, 24 GB, runs a 3B Q4 model at 10 to 15 tokens/s for $0.
- Qwen2.5-0.5B Q4_K_M is about 400 MB; Llama 3.1 8B Q4_K_M about 4.9 GB.
- Retry schedule: 200 ms base, doubling, 4 s cap, up to 50 percent jitter, 2 retries per backend
  by default.

## Ten likely questions

**1. Why not just use LiteLLM?** It is a more complete cloud proxy with budgets and keys. OpenInfer
is local-first (an in-process engine, hardware detection), treats privacy as a fail-closed
invariant, and is small enough to audit. If I needed 50 cloud providers and an admin UI today I
would use LiteLLM and put it behind the same policy idea.

**2. How do you know the privacy rule is not bypassed?** The flag is in code, the filter runs in
one function before any I/O, and two tests assert the cloud adapter is never called and
cloud-only chains return 422. A hosted OpenAI-compatible server must be configured as `cloud`;
there is deliberately no `local: true` override in config.

**3. How do you estimate cost before sending?** Characters divided by four for the prompt, plus
`max_tokens` at the output price. Pessimistic by design. The actual cost is recorded after, with
an `exact` flag when the backend reported real usage.

**4. What happens if a stream fails halfway?** The router is committed after the first chunk;
switching would splice two completions. The client gets an error event then `[DONE]`, the ledger
row says `error`. Before the first chunk, normal fallback applies.

**5. How would you scale this?** Stateless replicas behind a load balancer; move the ledger to
Postgres; vLLM for the GPU path; add a per-backend semaphore and a 429 with `Retry-After` for
back-pressure. The ledger already has the signals.

**6. Why SQLite?** Zero services for the first user, stdlib only, WAL for readers. Postgres is
the same SQL behind a URL. ADR-0004.

**7. Why YAML instead of code for policy?** Operators change routing; a PR diff is reviewable;
one file serves laptop, CI and prod because keyless backends go inactive. Trade-off: no runtime
reload yet.

**8. What about tokens for billing accuracy?** Backends that report usage are exact; llama.cpp
streams and some compatibility endpoints are estimated and flagged. I would not invoice
customers from estimates; I would from the provider's reported usage, which the ledger keeps.

**9. How do you test retries without sleeping?** `asyncio.sleep` is injected; tests pass a
recorder and assert the schedule: first delay in [100, 150) ms, second in [200, 300) ms.

**10. What would you build next?** Adaptive ordering from ledger p50 (data exists), a config
reload endpoint, per-tenant quotas keyed on tags, an NPU adapter via ONNX Runtime, and
idempotency keys for providers that support them.

# Routing policies and fallbacks

The router (`openinfer/router/engine.py`) is about 450 lines and is the heart of the project.
Read it with this lesson open.

## From a name to candidates

A request's `model` field is a routing handle. `Router.resolve()` tries three interpretations
in order:

1. **Alias**: a key under `aliases:` in `openinfer.yaml`. Its value is an ordered list of
   `backend/model` strings. Entries whose backend is not instantiated (disabled, or a cloud
   backend with no key) are skipped silently, so one file serves every environment.
2. **Qualified**: `ollama/llama3.1:8b`. Split on the first slash; if the prefix is a known
   backend, that is the single candidate. Model names containing slashes (OpenRouter's
   `meta-llama/...`) work because only the first slash is special.
3. **Bare**: `gpt-4o-mini`. Every backend that lists it becomes a candidate, in config order.

Anything else is `ModelNotFound` (404) with a message that lists the aliases. An empty model
falls back to `policy.default_alias`.

## Filtering: the policy

`Router.plan()` walks the candidates and drops those the request forbids, recording a reason
for each rejection (visible in the 422 message when nothing survives).

**Privacy.** If the hints contain `policy.private_tag` (default `private`), or the alias itself
is named that, every candidate whose backend has `local = False` is removed. `local` is a class
attribute of the adapter, not configuration, so it cannot be misconfigured. If nothing remains
the plan fails closed. ADR-0005 explains why.

**Cost ceiling.** `max_cost_usd` comes from the request header or body, capped by the global
policy value (the stricter wins). Before anything is sent, the router estimates cost as
`price(prompt_tokens_estimate, max_tokens or default_max_tokens)` using `pricing.yaml`. Local
backends cost zero unless given an amortised price. Estimates are deliberately pessimistic
(they assume the model fills `max_tokens`) because a ceiling that is sometimes exceeded is not
a ceiling.

**Latency ceiling.** `max_latency_ms` compares against the backend's measured p50 from the
ledger once there are `latency_min_samples` (5) successful rows, else against
`expected_latency_ms` from the config, else the candidate is kept (unknown is not slow). This is
the one place the ledger feeds back into routing, and it makes the system adaptive: a backend
that has been slow recently is skipped for latency-sensitive requests.

## Execution: retries and fallback

`_try_candidates()` iterates survivors in order. For each, it attempts up to `retries + 1`
times. The adapter's exception decides what happens:

- `BackendUnavailable` (connection refused, extra not installed): not retriable. Move on now.
- `RateLimited` (429), `BackendTimeout`, 5xx: retriable. Sleep then try the same backend.
- Other `BackendError` (400 bad request, unknown model): not retriable. Move on.
- Any other exception: logged as an adapter bug, treated as not retriable.

Backoff is `min(cap, base * 2^attempt) + uniform(0, base/2)`. The jitter prevents many
replicas retrying in lockstep against the same rate-limited provider. Tests inject a recording
`sleep` so they assert the schedule without waiting (`tests/conftest.py::NoSleep`).

If every candidate is exhausted, `AllBackendsFailed` (502) lists each attempt. One ledger row
with `status=error` is written for the request as a whole, not one per attempt, so error rates
in the ledger mean "requests that failed", which is what people ask about.

## Streaming changes one thing

For `chat_stream()`, fallback is only possible before the first chunk. The router awaits
`anext(stream)`; failures there follow the same retry/fallback rules. Once a chunk exists the
router calls `on_route(candidate)` and is committed: a mid-stream error ends the stream with an
error event, because switching backends would splice two different continuations together.
Token counts come from the final chunk's usage when the backend provides it, otherwise from an
estimate of the text, flagged `exact=False`.

## Designing a chain

Order by preference, not by reliability: the router handles reliability. A typical production
chain for `fast` is cheapest acceptable cloud first, then a second provider, then local, with
`echo` removed. For `private`, local engines only, best first. For `smart`, the best model you
can afford, with a cheaper fallback so an outage degrades rather than fails.

Two anti-patterns: a cloud backend inside the `private` alias (the tag will filter it, but the
alias then has fewer options than you think), and `echo/echo` left in production chains, which
turns outages into confident nonsense. Alert on `openinfer_requests_total{backend="echo"}`.

## Where to look

- `tests/test_router.py`: every rule above has a test; read them as executable documentation.
- `openinfer/server/schemas.py::parse_hints`: how headers, body and metadata become
  `RoutingHints`.

# 01. Architecture walkthrough

OpenInfer is a small system with five boxes. This lesson says why each exists and then follows
one request through all of them.

## The boxes

**Server** (`openinfer/server/`). A FastAPI app that speaks OpenAI's REST dialect. It exists so
clients do not change: the `openai` SDK, LangChain, curl. `app.py` builds the app and hangs the
other components on `app.state`; `schemas.py` holds the wire format; `routes/` has one module
per concern (chat, embeddings, audio, models, system). Nothing in here knows what a backend is
beyond the `Router` interface.

**Router** (`openinfer/router/engine.py`). The policy engine. It turns a model name into an
ordered list of `(backend, model)` candidates, removes the ones the request's constraints
forbid (privacy, cost, latency), and tries the rest with retries and backoff. It is the only
place that decides *where* a request runs. It exists so that decision is in one testable
function instead of scattered across clients.

**Backends** (`openinfer/backends/`). Adapters that translate our internal `ChatRequest` into
whatever an engine wants and the answer back into `ChatResult` or `ChatChunk`. `base.py` is the
interface; `echo.py` is deterministic for tests; `openai_compat.py` covers every OpenAI-dialect
server (vLLM, Groq, OpenRouter, Gemini, OpenAI) with `cloud.py` and `vllm.py` as thin presets;
`ollama.py` uses Ollama's native API for exact token counts; `llamacpp.py` runs a GGUF
in-process. Each declares `local` (does data leave the machine) and `pricing_provider`.

**Ledger and metrics** (`openinfer/ledger/`, `openinfer/metrics.py`). One row per request:
backend, model, tokens, cost, latency, time to first token, attempts, status, tags. SQLite by
default, Postgres optional. Prometheus counters and histograms are emitted alongside. The router
reads measured latency back from the ledger to enforce `max_latency_ms`, so the ledger is not
just reporting, it feeds routing.

**Hardware** (`openinfer/hardware/detect.py`). Probes CPU features and accelerators (CUDA via
`nvidia-smi`, ROCm via `rocm-smi`, Metal by platform, NPUs via OpenVINO or `/dev/accel`) and
recommends a llama.cpp build. It exists because "any hardware" is only true if the software can
tell you how to build for yours.

Supporting pieces: `config.py` (YAML policy with `${ENV}` expansion), `pricing.py` plus
`pricing.yaml` (dollars per million tokens), `bench/` (the benchmark), `cli.py` (Typer).

## One request, end to end

A client calls `POST /v1/chat/completions` with `{"model": "fast", "stream": true, ...}` and
the header `X-OpenInfer-Tags: private`.

1. **Auth middleware** in `app.py` checks the bearer token if `OPENINFER_API_KEY` is set.
2. **Validation.** `ChatCompletionRequest` (pydantic, `extra="allow"`) parses the body.
   `parse_hints()` merges the header, the optional `openinfer` body object and OpenAI
   `metadata` into a frozen `RoutingHints`. `to_internal()` yields a `ChatRequest` dataclass;
   from here on the OpenAI shape is gone.
3. **Plan.** `Router.plan()` calls `resolve("fast")`: it is an alias, so the chain from
   `openinfer.yaml` becomes candidates, skipping backends that are disabled or have no key.
   The `private` tag removes every candidate whose backend has `local = False`. Cost and latency
   ceilings would prune further. If nothing is left, `NoCandidates` becomes an HTTP 422 and no
   network call happens.
4. **Execute.** For streaming, `chat_stream()` opens the first candidate's stream and awaits
   the first chunk. A `BackendUnavailable` (connection refused) moves to the next candidate at
   once; a `RateLimited` or 5xx sleeps with exponential backoff and retries the same backend up
   to `policy.retries` times. Once a first chunk arrives the router is committed: it calls
   `on_route(candidate)` so the server can name the backend in the first SSE event.
5. **Stream.** `routes/chat.py` turns each `ChatChunk` into an OpenAI `chat.completion.chunk`
   JSON and writes `data: {...}\n\n`. The first event carries `delta.role`, the last carries
   `finish_reason`, an optional usage-only chunk follows if the client asked, then
   `data: [DONE]`.
6. **Record.** In a `finally`, the router builds a `LedgerEntry` with tokens (exact from the
   backend or estimated and flagged), cost from `Pricing`, latency and TTFT, and writes it to
   the ledger and metrics. `/v1/ledger` and `/metrics` read from there.

Non-streaming is the same path with `Router.chat()`, and the JSON response adds an `openinfer`
object plus `X-OpenInfer-*` headers so the client can see which backend actually served it.

## Why the internal types matter

The dataclasses in `openinfer/types.py` are the seam that keeps vendor quirks at the edges.
Ollama returns `arguments` as an object; OpenAI wants a JSON string: fixed in `ollama.py`.
Gemini rejects `stream_options`: a preset flag in `cloud.py`. DeepSeek puts thinking in a
separate field: `reasoning` on `ChatResult`. None of this touches the router or the server.
When you add a backend, you implement four methods and the rest of the system is unchanged.

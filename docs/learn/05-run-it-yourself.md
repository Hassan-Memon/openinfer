# 05. Run it yourself

Exercises in three flavours: break it (watch the failure handling), extend it (add a feature
along the existing seams), observe it (read the ledger and metrics). Each lists the files you
will touch and what success looks like. Start with `uv sync && uv run pytest` green.

## Break it

**1. Kill the local engine under a private request.** Start the server with Ollama *not*
running and send:

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -H 'X-OpenInfer-Tags: private' \
  -d '{"model":"private","messages":[{"role":"user","content":"secret"}]}' | jq .openinfer
```

You get echo with `attempts: 2`. Now remove `echo/echo` from the `private` alias in
`openinfer.yaml`, restart, repeat. Expect HTTP 422 and a message naming the refused candidates
in the ledger (`GET /v1/ledger`). That is fail closed. Put echo back.

**2. Simulate a rate limit.** In `tests/fakes.py` style, write a tiny ASGI app or use
`python -m http.server` tricks... simpler: point a `cloud` backend at a URL that returns 429.
A one-file FastAPI returning `{"error": {"message": "slow down"}}` with status 429 on
`/chat/completions` will do. Configure it as `type: cloud, base_url: http://localhost:9001,
api_key: x` and put it first in `fast`. Watch the logs: two retries with growing delays, then
fallback. Check `openinfer_fallbacks_total` on `/metrics`.

**3. Cut a stream.** Set `backends.echo.options.delay_ms: 300` so streams are slow, start a
streaming request with `curl -N`, and stop the fake backend (or Ollama) mid-stream. Confirm the
client receives an `error` event then `[DONE]`, and the ledger row has `status: error`.

**4. Break the config.** Set `OPENINFER_LEDGER_URL=postgresql://nobody@localhost/x` without the
extra. The server should fail at startup with a message naming `uv sync --extra postgres`, not
a traceback into psycopg internals.

## Extend it

**5. Add a backend.** Implement `openinfer/backends/lmstudio.py` as a preset over
`OpenAICompatBackend` with `local = True` and `base_url` defaulting to `http://localhost:1234/v1`,
register it in `backends/__init__.py::REGISTRY`, add a test with `httpx.MockTransport` in
`tests/test_backends_http.py`. Twenty lines. Then ask: should it be allowed in `private`? (Yes:
nothing leaves the machine.)

**6. Adaptive ordering.** In `Router.plan()`, after filtering, stable-sort candidates by
measured p50 from `self.ledger.latency_p50()` when a request carries a `latency` tag. Add a
test that records five slow rows for the first candidate and asserts it moves second. Keep the
default behaviour unchanged for requests without the tag.

**7. Config reload.** Add `POST /admin/reload` (behind the bearer token) that re-reads the YAML,
rebuilds backends and swaps `app.state.router`. Think about in-flight requests holding the old
router; that is fine, they finish on it.

**8. Ledger retention.** Add `openinfer ledger prune --days 30` to `cli.py` that deletes rows
older than the cutoff through a new `Ledger.prune(before_ts)` method implemented for SQLite
and Postgres. Test with the in-memory ledger.

**9. Back-pressure.** Add `max_concurrency` to `BackendConfig` and an `asyncio.Semaphore` in
`Backend` that wraps `chat` and `chat_stream`. When the semaphore cannot be acquired within
`policy.queue_timeout_ms`, raise `BackendError(retriable=True)` so the router backs off or
falls through. Observe TTFT in the ledger before and after under `hey` or `oha` load.

**10. Whisper.** `uv sync --extra whisper`, record a 10 second WAV, POST it to
`/v1/audio/transcriptions` with the `openai` SDK (`client.audio.transcriptions.create`). Then
add the segment timings to the ledger as `completion_tokens` equivalent and discuss whether
that is a sensible unit.

## Observe it

**11. Bench the real thing.** With Ollama running and a model pulled:

```bash
uv run openinfer bench --backend ollama --runs 5 --max-tokens 128
```

Compare TTFT and tokens/s against the echo row. Then set `max_tokens 512` and watch tokens/s
stay flat while total time grows: generation is per-token work.

**12. Prometheus in five minutes.** `uv sync --extra metrics`, run the server, then run a
Prometheus container with a scrape config pointing at `host.docker.internal:8000`. Query
`rate(openinfer_tokens_total[1m])` while you send traffic. Add Grafana if you like dashboards.

**13. Read the ledger like finance would.** After an hour of mixed traffic:

```bash
curl -s localhost:8000/v1/ledger?hours=1 | jq '.by_backend[] | {backend, model, requests, cost_usd, avg_latency_ms}'
```

Give your local backend an amortised price in `openinfer.yaml` (`pricing:`) and see how the
comparison changes. That number is the one that decides whether to buy a GPU.

**14. Hardware on three machines.** Run `openinfer hardware --json` on your laptop, a friend's
Mac and a cloud VM. Diff the `recommended.build` lines. That is the "any hardware" story in
three JSON blobs.

# Prometheus metrics

## The model

Prometheus is a pull-based time series database. Your service exposes a plain-text page
(`/metrics`) listing current values; a Prometheus server scrapes it every 15 to 60 seconds and
stores the samples; Grafana draws them. Nothing is pushed, nothing is buffered in your process
beyond the current counters, and if Prometheus is down your service does not care.

The text format looks like this (from `curl localhost:8000/metrics` after one request):

```
# HELP openinfer_requests_total Requests routed, by backend, model, endpoint and status
# TYPE openinfer_requests_total counter
openinfer_requests_total{backend="echo",endpoint="chat",model="echo",status="ok"} 1.0
openinfer_request_latency_seconds_bucket{backend="echo",endpoint="chat",le="0.05"} 1.0
openinfer_request_latency_seconds_bucket{backend="echo",endpoint="chat",le="0.1"} 1.0
...
openinfer_request_latency_seconds_sum{backend="echo",endpoint="chat"} 0.0013
openinfer_request_latency_seconds_count{backend="echo",endpoint="chat"} 1.0
```

## Metric types

**Counter**: only goes up (resets on restart). Requests, tokens, dollars. You never read a
counter's value directly; you ask Prometheus for its rate: `rate(openinfer_requests_total[5m])`
is requests per second over the last five minutes.

**Gauge**: goes up and down. Queue depth, loaded models, temperature. OpenInfer has none yet;
"requests in flight" would be the first to add.

**Histogram**: counts observations into cumulative buckets (`le` = less than or equal) plus a
sum and a count. From buckets Prometheus computes quantiles across time and across instances:
`histogram_quantile(0.95, sum by (le, backend) (rate(openinfer_request_latency_seconds_bucket[5m])))`
is p95 latency per backend. Bucket boundaries are fixed at creation; the ones in
`openinfer/metrics.py` span 50 ms to 120 s because an LLM request can take either.

**Summary** computes quantiles in the client. Avoid it: quantiles cannot be aggregated across
instances. Histograms can.

## Labels

Every metric carries labels (`backend`, `model`, `endpoint`, `status`), and each distinct label
combination is a separate time series. That is the power (slice by anything) and the hazard
(cardinality). Never put user ids, request ids or prompt text in a label. Model names are fine
because there are dozens, not millions.

## What OpenInfer exposes

Defined in `openinfer/metrics.py`, updated from the same `LedgerEntry` the ledger writes, so
the two never disagree:

| Metric | Type | Labels | Question it answers |
|---|---|---|---|
| `openinfer_requests_total` | counter | backend, model, endpoint, status | traffic and error rate per backend |
| `openinfer_tokens_total` | counter | backend, model, direction | token volume, prompt vs completion |
| `openinfer_cost_usd_total` | counter | backend, model | spend rate: `rate(...[1h]) * 3600` is dollars per hour |
| `openinfer_request_latency_seconds` | histogram | backend, endpoint | p50/p95/p99 latency |
| `openinfer_ttft_seconds` | histogram | backend | time to first token for streams |
| `openinfer_fallbacks_total` | counter | requested_model | how often the first choice failed |

Useful alerts: error ratio above 5 percent for 10 minutes
(`sum(rate(...{status="error"}[10m])) / sum(rate(...[10m])) > 0.05`), any traffic on
`backend="echo"` in production, p95 TTFT above 2 s on the `private` path.

## Implementation notes

`prometheus_client` is an optional extra. Without it `Metrics.enabled` is false, every
`observe()` is a no-op and `/metrics` returns 501 with an install hint, so the core has no
hard dependency. Each `Metrics` instance owns its own `CollectorRegistry` instead of the
library's global one; otherwise creating two apps in one test process raises "duplicate
timeseries". `generate_latest(registry)` renders the page.

Metrics are not a replacement for the ledger: they answer "how many and how fast, right now and
over time" cheaply; the ledger answers "which request, what did it cost, which tags". Use both.

## Where to look

- `openinfer/metrics.py`, `openinfer/server/routes/system.py::metrics`.
- `tests/test_metrics_audio.py`: asserts the exact exposition line.
- Grafana: import a dashboard with the three queries above and you have observability in an
  afternoon.

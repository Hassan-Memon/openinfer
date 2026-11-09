# Cost accounting for LLMs

## Tokens, not requests

LLM providers bill per token, with separate prices for input (prompt) and output (completion)
tokens, quoted per million. Output is 2 to 5 times more expensive than input because generating
a token requires a full forward pass while prompt tokens are processed in one batched pass.
Prices are small numbers with many zeros: Groq's 8B endpoint is $0.05 per million input tokens,
$0.08 per million output; GPT-4o is $2.50 and $10.00. `openinfer/pricing.yaml` holds a table of
these, keyed by provider and model, and `Pricing.cost()` does the arithmetic:

```
cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
```

A token is roughly four characters of English, less for code and other languages, more for
whitespace-heavy text. Exact counts come from the tokenizer, which only the backend has. When a
backend reports `usage`, the ledger stores it with `usage_exact = 1`. When it does not
(llama.cpp streams, some compatibility endpoints), `estimate_tokens()` in `openinfer/types.py`
divides characters by four and the row is flagged `usage_exact = 0`. Never present an estimate
as a fact; the flag is there so totals can be caveated.

## Estimating before sending

The router enforces `max_cost_usd` before a request goes out, so it must estimate. It uses the
character-based prompt estimate plus `max_tokens` (or `policy.default_max_tokens`, 512) at the
output price. That is a worst case: most completions are shorter than `max_tokens`. A ceiling
should be a ceiling, so pessimism is correct here; the actual cost recorded afterwards is
usually lower.

## Local is not free, it is amortised

An Ollama or llama.cpp backend has zero marginal cost per token, which is what the pricing
table says by default. But the machine costs something: electricity on a laptop, $100 to
$1,000 a month on a cloud instance. To compare fairly, give a local backend an amortised price
in `openinfer.yaml`:

```yaml
backends:
  vllm:
    pricing:
      meta-llama/Llama-3.1-8B-Instruct: { input: 0.25, output: 0.25 }
```

The method: instance $/hour divided by sustained tokens/hour at your real utilisation. The AWS
cost model (`docs/aws-cost-model.md`) works several examples; the punchline is that a GPU at 5
percent utilisation is far more expensive per token than any API, and at 60 percent it is
cheaper than all of them. Only the ledger can tell you which one you are.

## What the ledger records

One row per request (`openinfer/ledger/base.py::LedgerEntry`): endpoint, requested model and
alias, backend and concrete model, status, prompt and completion tokens, whether exact, cost,
latency, time to first token, attempts, tags, error. No prompt or completion text.

`GET /v1/ledger?hours=24` aggregates: totals (requests, errors, tokens, cost, p50/p95 latency,
p50 TTFT) and a breakdown by backend and model. `openinfer bench` adds dollars per million
output tokens next to measured tokens per second so a table row answers "how fast and how much"
for each backend.

## Latency is a cost too

Two numbers matter. **Time to first token (TTFT)** is what a user perceives as responsiveness
and is dominated by prompt processing (prefill) plus network and queueing. **Tokens per second**
after the first token is reading speed; above about 20 tokens/s people stop noticing. The bench
measures both from a stream: TTFT is the time until the first content (or reasoning) delta,
throughput is completion tokens divided by the time after that delta.

The router turns latency into a routing input: `max_latency_ms` compares the ledger's measured
p50 for each candidate and skips the slow ones.

## Why percentiles, not averages

A mean latency hides a bimodal distribution (fast cache hits and slow cold starts). p50 tells
you the typical experience, p95 tells you what one in twenty users sees. The ledger computes
them in Python over the last 10,000 rows (`percentile()` in `ledger/base.py`) so SQLite and
Postgres agree, and Prometheus histograms give the same information over time windows.

## Keeping the table honest

Prices change. `pricing.yaml` carries a "last reviewed" note and the README says so. Unknown
models get `known: false` and zero cost rather than a guess; the bench prints "unknown, assumed
0" for them. Per-backend `pricing:` overrides win over the table, so a team can pin negotiated
rates without editing package files.

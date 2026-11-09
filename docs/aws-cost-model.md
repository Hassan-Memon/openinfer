# AWS EC2 cost model (documented, not required)

OpenInfer runs on free tiers; nothing here is needed to use it. This page exists because the
first question in any platform conversation is "what would this cost if we did pay for it", and
because the ledger makes the comparison concrete: it records tokens and seconds per backend, so
you can divide an instance's hourly price by its measured throughput.

Prices are **on-demand, us-east-1, Linux, November 2026 list prices rounded**, and will drift.
Treat the table as a method, not a quote. Spot pricing is typically 60 to 70 percent lower for
the GPU instances; Savings Plans give 30 to 40 percent off for 1 year commitments.

## Instance options

| Instance | Hardware | $/hour | $/month (730 h) | Fits | Rough tokens/s (llama.cpp or vLLM, Q4/FP16) | Notes |
|---|---|---:|---:|---|---|---|
| t4g.small (free tier 750 h/mo for 12 months) | 2 vCPU Graviton2, 2 GB | 0.017 | 12 | 0.5B Q4 | 3 to 6 | API gateway role; model fits but barely |
| t4g.medium | 2 vCPU Graviton2, 4 GB | 0.034 | 25 | 1.5B Q4 | 5 to 8 | Burstable; sustained inference exhausts credits |
| c7g.xlarge | 4 vCPU Graviton3, 8 GB | 0.145 | 106 | 3B Q4 | 10 to 15 | Same class as Oracle's free A1 but Graviton3 is faster per core |
| c7g.4xlarge | 16 vCPU Graviton3, 32 GB | 0.58 | 423 | 8B Q4 | 15 to 25 | CPU ceiling for interactive 8B |
| g4dn.xlarge | 1x T4 16 GB, 4 vCPU | 0.526 | 384 | 8B Q4, 3B FP16 | 25 to 40 | Cheapest NVIDIA GPU; good for llama.cpp CUDA offload |
| g5.xlarge | 1x A10G 24 GB, 4 vCPU | 1.006 | 734 | 8B FP16, 13B Q4 | 60 to 90 single stream; vLLM batches far higher aggregate | Standard "one GPU" vLLM box |
| g6.xlarge | 1x L4 24 GB, 4 vCPU | 0.805 | 588 | 8B FP16 | 50 to 80 | Newer than g5 at lower price; availability varies |
| g6e.xlarge | 1x L40S 48 GB, 4 vCPU | 1.861 | 1,359 | 70B Q4 (tight), 8B at high concurrency | 100+ single stream on 8B | For a 70B class model this is the entry point |
| p4d.24xlarge | 8x A100 40 GB | 32.77 | 23,922 | 70B FP16 with TP | n/a | Shown for scale only |
| inf2.xlarge | 1x Inferentia2 | 0.758 | 553 | 8B via Neuron compile | comparable to A10G | Not CUDA; proves the "any hardware" point but needs model compilation |

Throughput numbers are the public ranges people report for llama.cpp and vLLM on these parts and
depend heavily on quantisation, prompt length and batch size. Measure your own with
`openinfer bench` before believing any of them.

## Cost per million output tokens

Hourly price divided by tokens per hour. Single stream (no batching) is pessimistic for GPUs,
which is exactly the case llama.cpp in-process optimises for; vLLM with 16 to 64 concurrent
requests gets 5 to 20x more aggregate tokens from the same GPU.

| Setup | $/hour | tokens/s (single) | $/1M tokens (single stream) | $/1M tokens (vLLM batched, estimate) |
|---|---:|---:|---:|---:|
| c7g.xlarge, 3B Q4 CPU | 0.145 | 12 | 3.36 | n/a |
| g4dn.xlarge, 8B Q4 CUDA | 0.526 | 30 | 4.87 | 0.50 |
| g5.xlarge, 8B FP16 | 1.006 | 75 | 3.73 | 0.25 |
| g6e.xlarge, 8B FP16 | 1.861 | 110 | 4.70 | 0.20 |
| Groq llama-3.1-8b-instant (from `openinfer/pricing.yaml`) | | | 0.08 | |
| OpenAI gpt-4o-mini | | | 0.60 | |

Formula: `$/1M = price_per_hour / (tokens_per_s * 3600) * 1_000_000`.

## What the table says

1. **Per token, a single self-hosted GPU is 10 to 60x more expensive than Groq's 8B endpoint** unless it is kept busy with batching. Self-hosting wins on privacy, latency variance and control, not on unit price at low volume.
2. **CPU inference on Graviton is a real option for small models.** c7g.xlarge at $106/month runs a 3B model all day; Oracle gives you the same class for free, which is why the Oracle guide is the reference "private" deployment.
3. **Utilisation decides everything.** A $734/month g5 serving 100 requests a day costs more per token than any API. The same box at 50 percent utilisation with vLLM is cheaper than every API in the table. The ledger's `by_backend` view tells you which side of that line you are on.
4. **The routing policy is the cost control.** `private` goes to the box you already pay for, `smart` goes to the best model at list price, `fast` goes wherever is cheapest right now. The ceiling (`max_cost_usd`) stops any single request from being expensive by accident.

## How to use this with OpenInfer

Give a self-hosted backend an amortised price so it shows up in the ledger honestly:

```yaml
backends:
  vllm:
    type: vllm
    base_url: http://10.0.0.5:8000/v1
    models: [meta-llama/Llama-3.1-8B-Instruct]
    pricing:
      meta-llama/Llama-3.1-8B-Instruct: { input: 0.25, output: 0.25 }   # g5 at ~50% utilisation
```

Then `GET /v1/ledger` and `openinfer bench` compare it against cloud on equal terms, and
`max_cost_usd` applies to it like any other candidate.

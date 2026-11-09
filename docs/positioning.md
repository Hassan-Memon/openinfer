# Positioning

## Who it is for

1. **Teams with mixed data sensitivity.** A clinic, law firm or bank wants frontier-model
   quality for general work and a hard guarantee that certain requests never leave their
   machines. OpenInfer gives both behind one URL, with the privacy rule enforced in code.
2. **Developers tired of CUDA lock-in.** You have a MacBook, a colleague has a Windows laptop
   with an NVIDIA card, the CI box is an ARM VM, production is an AMD instance. One GGUF, one
   config, one API across all of them.
3. **Platform engineers who need a cost ledger before finance asks.** Every request is
   priced, timed and attributed. `/v1/ledger`, `/metrics` and `openinfer bench` turn "what does
   inference cost us" into a query.
4. **Free-tier builders.** Groq, Gemini, OpenRouter and Ollama each have limits and outages.
   Alias chains with backoff make a set of free tiers behave like one reliable endpoint.

## Who it is not for (yet)

- High-concurrency serving on one big GPU. Use vLLM directly (or behind OpenInfer as a backend).
- Fine-tuning, training, model management UIs.
- Teams that need per-tenant policies, quotas and billing today. The hooks exist (tags, ledger),
  the product features do not.

## Alternatives, honestly

### NVIDIA NIM and the CUDA stack

NIM packages optimised inference microservices (TensorRT-LLM, Triton) as containers with an
OpenAI-compatible API. On NVIDIA hardware it will be faster than llama.cpp on the same GPU and
comes with enterprise support. It is also NVIDIA-only by construction and licensed through
NVIDIA AI Enterprise for production use. OpenInfer runs the same model on CPU, Metal, ROCm and
NVIDIA, and routes to NIM or vLLM when you do have the GPU. Choose NIM if you are all-NVIDIA
and need maximum throughput; choose OpenInfer if hardware diversity, privacy routing or cost
visibility are the problem.

### Apple on-device (Apple Intelligence, Foundation Models framework, MLX)

Apple's stack is excellent on Apple silicon and private by design, but it is Apple-only and
the system models are not something you pick, swap or expose over HTTP to a Linux server.
MLX is a good engine for Macs. OpenInfer uses Metal through llama.cpp today and treats a Mac
as one more local backend; it does not replace the OS features.

### The OpenAI API (and Anthropic, Google, Groq direct)

Best models, simplest start, no hardware. Also: every token leaves your network, pricing is
set by the vendor, and outages are yours to absorb. OpenInfer keeps the exact same API so you
can start on OpenAI and move individual aliases local later without touching clients. Most
users will keep a cloud backend in the chain; the point is choosing per request.

### Ollama

Ollama is the easiest way to run local models and has an OpenAI-compatible endpoint. It is a
backend in OpenInfer, not a competitor: OpenInfer adds routing across many backends (including
several Ollama hosts), privacy enforcement, cost and latency ledgers, cloud fallbacks and a
bench. If you only ever need one local model on one machine, use Ollama alone.

### LiteLLM

LiteLLM is a mature proxy over 100+ cloud providers with budgets, virtual keys and a UI. It is
the closest comparison for the routing layer and is more complete there. OpenInfer differs in
three ways: it is local-first (an in-process llama.cpp engine, hardware detection, a real
"any hardware" story) rather than a cloud proxy; privacy is a code-level routing invariant
with fail-closed semantics; and it is small enough to read in an afternoon (one package, no
database required). If you need enterprise key management across 50 cloud providers today,
LiteLLM. If you need one binary that runs a private model on whatever box you have and falls
back to free cloud tiers, OpenInfer.

### vLLM (and SGLang, TGI)

Server-grade engines with continuous batching. They are the right choice when a GPU is kept
busy by many concurrent users, and OpenInfer speaks to them over HTTP as the `vllm` backend.
They do not route, do not do privacy policy, and are heavy to install on a laptop.

## Why this wins, in one paragraph

Nobody else combines the four things in one small, readable package: an OpenAI-compatible
front door, a local engine that runs the same quantised model on every accelerator, a router
that enforces privacy and cost per request and fails closed, and a ledger that tells you what
it all cost. Each piece exists elsewhere; the combination is what a regulated team actually
needs on day one, and it runs on free tiers end to end.

## Numbers to quote

From this repo on a CPU-only laptop (see README bench table and `docs/aws-cost-model.md`):
a 1.5B model on CPU gives roughly 380 ms time to first token and 16 tokens/s; Groq's 8B
endpoint is about $0.08 per million output tokens versus $3 to $5 for a single-stream
self-hosted GPU; Oracle's free ARM tier runs a 3B model at 10 to 15 tokens/s for $0.

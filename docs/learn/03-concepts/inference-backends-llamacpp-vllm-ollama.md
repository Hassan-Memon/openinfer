# Inference backends: llama.cpp, vLLM, Ollama

An inference engine loads model weights and turns a prompt into tokens. The three in this repo
make different trade-offs, and the router lets you use all of them.

## llama.cpp

A C/C++ project by Georgi Gerganov that started as "run LLaMA on a MacBook" and became the most
portable LLM engine. Its tensor library (ggml) has compute backends for CPU (AVX2, AVX-512, ARM
NEON), Apple Metal, NVIDIA CUDA, AMD HIP (ROCm), Vulkan and Intel SYCL. It reads GGUF files,
which carry quantised weights (see the quantization lesson), and can split layers between GPU
and CPU with one parameter, `n_gpu_layers`. `-1` means "as many as fit".

Strengths: runs anywhere, tiny memory footprint, fast single-stream generation, no Python or
PyTorch required. Weaknesses: batching across many concurrent users is limited, and the Python
binding (`llama-cpp-python`) compiles the C++ at install time, which is why it is an optional
extra here and why `openinfer hardware` prints the right `CMAKE_ARGS`.

In OpenInfer: `openinfer/backends/llamacpp.py`. The model loads lazily on first request, calls
run in a worker thread (`asyncio.to_thread`) behind a lock because one `Llama` instance is not
thread safe, and streaming bridges the synchronous generator to the event loop through a queue.
`create_chat_completion()` returns OpenAI-shaped dicts, so the parser from `openai_compat.py` is
reused.

## vLLM

A Python serving engine from UC Berkeley built around PagedAttention: the KV cache (the
attention state for every token generated so far) is stored in fixed-size blocks like virtual
memory pages, so many requests share GPU memory without fragmentation. Combined with continuous
batching (new requests join a batch between decode steps instead of waiting for the batch to
finish) it gets 5 to 20x the aggregate throughput of a single-stream engine on the same GPU.

Strengths: throughput at concurrency, tensor parallelism across GPUs, an OpenAI-compatible
server out of the box. Weaknesses: NVIDIA first (AMD and CPU support exist and are improving),
several GB of dependencies, needs a real GPU to make sense, quantisation support is
GPU-specific (AWQ, GPTQ, FP8) rather than GGUF.

In OpenInfer: `openinfer/backends/vllm.py` is a 20 line preset over `OpenAICompatBackend` with
`local = True` and no pricing provider. You run vLLM yourself and give its URL.

## Ollama

A Go application that wraps llama.cpp with model management (`ollama pull llama3.1`), a model
library with sensible default quantisations, automatic GPU detection, request queueing and a
REST API. It is the easiest way to run a local model and is what most developers already have
on their laptop.

Strengths: one command install, handles builds for Metal/CUDA/ROCm, keeps models loaded and
unloads them after idle, runs as a background service. Weaknesses: another process to run, one
model loaded at a time by default, throughput is llama.cpp's.

In OpenInfer: `openinfer/backends/ollama.py` uses the native `/api/chat` and `/api/embed`
endpoints rather than Ollama's `/v1` shim because the native responses include exact
`prompt_eval_count` and `eval_count`, which the ledger and bench want. Streaming is
newline-delimited JSON. Tool call arguments come back as objects and are re-encoded as JSON
strings for OpenAI clients. Thinking from reasoning models arrives in `message.thinking` and is
surfaced as `reasoning_content`.

## Choosing

| Situation | Backend |
|---|---|
| Laptop, one user, want the smallest footprint | llama.cpp in-process |
| Laptop, want zero build hassle | Ollama |
| One GPU, many concurrent users | vLLM |
| ARM VM with no GPU (Oracle free tier) | llama.cpp (NEON) or Ollama |
| Need a model nobody has quantised to GGUF | vLLM or a cloud backend |

The router does not need you to pick. A chain like `[vllm/..., ollama/..., llamacpp/...]`
uses the server-grade engine when it is up and degrades to the laptop engine when it is not.

## Where to look

- `openinfer/backends/base.py`: the four methods every engine implements.
- `tests/test_backends_http.py`: Ollama and OpenAI-dialect wire formats, mocked.
- ADR-0003 for why llama.cpp came first.

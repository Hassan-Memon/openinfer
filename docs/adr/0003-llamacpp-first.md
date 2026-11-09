# ADR-0003: llama.cpp (GGUF) as the first local engine, vLLM and Ollama over HTTP

- Status: accepted
- Date: 2026-11-09
- Deciders: Hassan Memon

## Context and problem statement

The pitch is "the same model on CPU, Metal, CUDA, ROCm or NPU". Which engine do we integrate
in-process first? Candidates: llama.cpp via `llama-cpp-python`, vLLM, Hugging Face
`transformers` with PyTorch, ONNX Runtime, MLX (Apple only), OpenVINO (Intel).

## Decision drivers

- Hardware breadth: one artefact must run on a laptop CPU, an M-series Mac, an NVIDIA box, an
  AMD box and the ARM VMs free tiers give away.
- Memory: free and consumer hardware has 2 to 24 GB. Quantised weights are a requirement, not an
  optimisation.
- Dependency weight: PyTorch plus CUDA is 2 to 5 GB of wheels and a GPU driver matrix; CI must
  stay fast and the Docker image small.
- Throughput at concurrency matters for servers, but the first user is a developer with one
  machine.

## Considered options

1. **llama.cpp (GGUF)**: C++ engine with backends for CPU (AVX2, AVX-512, NEON), Metal, CUDA,
   HIP/ROCm, Vulkan, SYCL. Quantisation formats from 2 to 8 bit in one file format. Python binding
   compiles for the target at install time. Single-stream focus, modest batching.
2. **vLLM**: best-in-class batched throughput (PagedAttention, continuous batching) on NVIDIA,
   growing AMD and CPU support. Heavy install, GPU-first, unquantised or GPU-specific quants.
3. **transformers + PyTorch**: maximum model coverage, poor CPU performance, huge footprint.
4. **ONNX Runtime / OpenVINO**: good on Intel CPU and NPU, model conversion step, narrower LLM
   coverage.

## Decision

Option 1 in-process behind the optional `llamacpp` extra, with vLLM and Ollama (itself a
llama.cpp wrapper with model management) integrated as HTTP backends. The default local model is
Qwen2.5-0.5B-Instruct Q4_K_M (about 400 MB), small enough for CI machines and Hugging Face's
free CPU tier, with `openinfer download-model` fetching it from Hugging Face.

`openinfer hardware` detects the accelerator and prints the `CMAKE_ARGS` that make the same
wheel build for Metal, CUDA or ROCm (`openinfer/hardware/detect.py`). The extra stays optional
because the wheel compiles from source on many platforms (Windows needs MSVC build tools), and
CI runs on the echo backend.

## Consequences

- Positive: the "any hardware" claim is literally one file and one flag (`n_gpu_layers`).
- Positive: Ollama users get the same engine with none of the build hassle; vLLM users get
  server-grade batching when they have the GPU for it.
- Negative: single-process llama.cpp serialises requests (a lock in `backends/llamacpp.py`);
  under concurrency the right move is Ollama or vLLM, which the policy makes a one-line change.
- Negative: NPUs are detected but not used by llama.cpp; OpenVINO or ONNX Runtime (QNN) exports
  are the path there and the recommendation says so rather than pretending.
- Negative: the llama.cpp path is not exercised in CI; it is verified manually and documented as
  such in the README status section.

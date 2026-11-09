# Learn OpenInfer

Short lessons, each 300 to 700 words, written for a Python developer who knows FastAPI and
wants to understand inference infrastructure well enough to defend the design in an interview.

1. [Architecture walkthrough](01-architecture-walkthrough.md): why each box exists and the request path end to end
2. [System design trade-offs](02-system-design-tradeoffs.md): scaling, failure modes, back-pressure, idempotency, what breaks first at 10x
3. Concepts (one file each):
   - [OpenAI-compatible APIs and SSE](03-concepts/openai-compatible-apis-and-sse.md)
   - [Inference backends: llama.cpp, vLLM, Ollama](03-concepts/inference-backends-llamacpp-vllm-ollama.md)
   - [Quantization and GGUF](03-concepts/quantization-and-gguf.md)
   - [Hardware accelerators: CPU, Metal, CUDA, ROCm, NPU](03-concepts/hardware-accelerators-cpu-metal-cuda-rocm-npu.md)
   - [Routing policies and fallbacks](03-concepts/routing-policies-and-fallbacks.md)
   - [Cost accounting for LLMs](03-concepts/cost-accounting-for-llms.md)
   - [Prometheus metrics](03-concepts/prometheus-metrics.md)
4. [Interview talk track](04-interview-talk-track.md): 2 minute and 15 minute versions, numbers, 10 likely questions
5. [Run it yourself](05-run-it-yourself.md): break it, extend it, observe it

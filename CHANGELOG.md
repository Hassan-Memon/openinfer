# Changelog

All notable changes to OpenInfer are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-10

First public release. This repository was created empty in 2020 and is now OpenInfer.

### Added
- **OpenAI-compatible server (FastAPI):** `/v1/chat/completions` with streaming over
  server-sent events, `/v1/embeddings` in float or base64, `/v1/models`,
  `/v1/audio/transcriptions` with faster-whisper as an optional extra, `/health`,
  `/v1/hardware`, `/v1/ledger` and Prometheus `/metrics`. The official `openai` Python SDK is
  tested against it.
- **Backends:** echo for tests and demos, Ollama, vLLM, llama.cpp as an optional extra, and
  OpenAI-compatible cloud presets for Groq, OpenRouter, Gemini and OpenAI.
- **Router:** aliases (`fast`, `smart`, `private`, `embed`) defined in `openinfer.yaml`, a
  privacy tag that restricts a request to local backends and never falls through to the cloud,
  cost and latency ceilings, retries with exponential backoff, and fallback through each alias
  chain.
- **Ledger:** every request records tokens, cost, latency and time to first token in SQLite by
  default, or Postgres.
- **Hardware detection:** CPU features, NVIDIA CUDA, AMD ROCm, Apple Metal and NPUs, with the
  recommended llama.cpp build flags.
- **CLI:** `openinfer serve`, `bench`, `hardware`, `models`, `config` and `download-model`.
- **Deployment:** a multi-stage Docker image that runs as a non-root user, Docker Compose with an
  optional Ollama profile, and deploy guides for Hugging Face Spaces, Render and Oracle Cloud
  Always Free, plus an AWS EC2 cost model.
- **Landing page:** a static Cloudflare Worker in `site/` with an in-browser router playground,
  live at https://openinfer.hassan-memon.workers.dev.
- **Documentation:** six architecture decision records, positioning against NVIDIA NIM, Apple
  on-device models, the OpenAI API, Ollama, LiteLLM and vLLM, and a teaching track in
  `docs/learn/`.
- **Tests:** 111 offline tests, plus CI on Python 3.12 and 3.13 with a Docker smoke test.

### Fixed
- Reasoning output from thinking models, such as DeepSeek R1 through Ollama, is returned as
  `reasoning_content`, and the benchmark counts its first token.
- Embeddings support `encoding_format=base64`, which the OpenAI SDK requests by default.
- Environment values that are unsafe in YAML, such as `sqlite:///:memory:`, no longer break
  config parsing.
- The quickstart installs from GitHub, because the package is not published on PyPI.

### Measured
- On a laptop CPU with no GPU, `deepseek-r1:1.5b` through Ollama reached a median time to first
  token of 380 ms and 16 generated tokens per second.

[0.1.0]: https://github.com/Hassan-Memon/openinfer/commits/main

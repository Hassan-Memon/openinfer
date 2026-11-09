# Deploy to Hugging Face Spaces (Docker, CPU, free)

Spaces give you a free 2 vCPU / 16 GB container with a public HTTPS URL. It sleeps after 48
hours without traffic and wakes on the next request. That is enough for a live demo of the API
on the echo backend, or a tiny real model (Qwen2.5-0.5B Q4_K_M at roughly 5 to 10 tokens/s).

## 1. Create the Space

1. huggingface.co, New Space, SDK: **Docker**, hardware: **CPU basic (free)**.
2. Clone the Space repo and copy this project into it (or add the Space as a second git remote).

## 2. Port and config

Spaces route traffic to port **7860**. The image reads `OPENINFER_PORT`, so set it in the
Space's `README.md` front matter and variables:

```yaml
---
title: OpenInfer
emoji: ⚡
sdk: docker
app_port: 7860
---
```

In Space settings, Variables: `OPENINFER_PORT=7860`.
Secrets (optional): `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENINFER_API_KEY`.
Only backends whose key is present become active, so with no secrets the Space serves `echo`.

## 3. Ledger storage

The container filesystem is ephemeral. The default `OPENINFER_LEDGER_URL` in the image is
`sqlite:////data/openinfer.db`, which the Dockerfile creates. Stats survive restarts only if you
enable persistent storage (paid) or point the ledger at a free Postgres (Neon or Supabase):

```
OPENINFER_LEDGER_URL=postgresql://user:pass@ep-xxx.neon.tech/openinfer?sslmode=require
```

and build with `--extra postgres` (edit the two `uv sync` lines in the Dockerfile).

## 4. Optional: a real model on CPU

Add to the Dockerfile builder stage:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra metrics --extra llamacpp
RUN uv run openinfer download-model --dest /app/models
```

and copy `/app/models` into the runtime stage. Set `LLAMACPP_MODEL_PATH=/app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
and flip `backends.llamacpp.enabled: true` in `openinfer.yaml`. Expect a 10 minute first build
(llama.cpp compiles from source) and 1 to 2 GB image.

## 5. Verify

```bash
curl https://<user>-<space>.hf.space/health
curl https://<user>-<space>.hf.space/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"fast","messages":[{"role":"user","content":"hello"}]}'
```

## Limits to know

- 2 vCPU: a 0.5B model is interactive, a 3B model is not. Route anything heavier to a cloud alias.
- Sleeping Spaces take 30 to 60 s to wake; the first request will time out on strict clients.
- Set `OPENINFER_API_KEY` if you add any cloud keys. A public URL with a Groq key and no auth is a free proxy for strangers.

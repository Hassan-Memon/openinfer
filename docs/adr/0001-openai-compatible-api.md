# ADR-0001: Expose an OpenAI-compatible API rather than a native one

- Status: accepted
- Date: 2026-11-09
- Deciders: Hassan Memon

## Context and problem statement

OpenInfer sits between applications and many inference engines. Applications need one stable
contract. We could design our own request and response schema (cleaner, fits our routing hints
naturally) or adopt the OpenAI REST dialect (`/v1/chat/completions`, `/v1/embeddings`, SSE
chunks ending in `data: [DONE]`) that most of the ecosystem already speaks.

## Decision drivers

- Zero client changes: the `openai` SDKs, LangChain `ChatOpenAI`, LlamaIndex, Vercel AI SDK,
  Continue, Open WebUI and most agent frameworks only need a `base_url`.
- Backends already speak it: vLLM, llama.cpp server, LM Studio, Groq, OpenRouter, Gemini's
  compatibility endpoint and OpenAI itself. One adapter (`openinfer/backends/openai_compat.py`)
  covers most of them.
- Our extras (routing tags, cost ceilings, which backend served) must be expressible without
  breaking strict clients.

## Considered options

1. **Native API** with first-class routing fields. Clean but every client needs a shim.
2. **OpenAI-compatible API** with extensions in headers and an `openinfer` object.
3. **Both** (native plus a compatibility layer). Twice the surface to test and document.

## Decision

Option 2. The wire format is OpenAI's, validated loosely (`extra="allow"`) so unknown fields
pass through to backends that understand them. Extensions live in three places that strict
clients ignore: request headers (`X-OpenInfer-Tags`, `X-OpenInfer-Max-Cost-USD`,
`X-OpenInfer-Max-Latency-Ms`), an optional `openinfer` object in the body, and OpenAI's own
`metadata` field. Responses add an `openinfer` object and `X-OpenInfer-*` headers.

Model names are the routing handle: an alias (`fast`), a qualified `backend/model`, or a bare
model name a backend lists. This keeps "pick the policy" inside the field every client already
has.

Internally the server translates to small dataclasses (`openinfer/types.py`) so the router and
adapters never touch the wire format. A test suite drives the real `openai` SDK through the ASGI
app (`tests/test_openai_sdk.py`) to keep the shapes honest.

## Consequences

- Positive: adoption cost is a URL change; every example in the OpenAI docs works.
- Positive: new backends are mostly configuration if they are OpenAI-compatible.
- Negative: we inherit OpenAI's quirks (base64 embeddings by default in the SDK, `n`, `logprobs`,
  `stream_options`) and must either implement or reject them clearly (we return 400 for `n > 1`).
- Negative: features without an OpenAI analogue (reasoning text, routing metadata) ride on
  conventions (`reasoning_content`, `openinfer` object) rather than a schema we control.
- Neutral: the API version is pinned to behaviour, not to OpenAI's changelog; we add fields when
  clients need them.

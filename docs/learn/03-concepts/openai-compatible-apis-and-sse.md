# OpenAI-compatible APIs and SSE

## The de facto standard

OpenAI's REST API became the lingua franca of LLM serving not because it is elegant but because
it shipped first and every client library targets it. "OpenAI-compatible" means a server
accepts the same JSON at the same paths and returns the same shapes, so a client only changes
`base_url`. vLLM, llama.cpp's server, Ollama, LM Studio, Groq, OpenRouter, Together, Mistral and
Google's Gemini compatibility endpoint all do this. OpenInfer does it on the front
(`openinfer/server/`) and consumes it on the back (`openinfer/backends/openai_compat.py`).

The core request is `POST /v1/chat/completions`:

```json
{"model": "fast", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2}
```

and the response:

```json
{"id": "chatcmpl-...", "object": "chat.completion", "created": 1700000000, "model": "echo",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}}
```

Fields that matter for compatibility: `choices[0].message.content` (may be `null` when
`tool_calls` is present), `finish_reason` (`stop`, `length`, `tool_calls`), and `usage`. The
official SDKs parse these strictly enough that a missing `object` or a string where an int is
expected breaks them. `tests/test_openai_sdk.py` drives the real SDK against the app to catch
exactly that. Extra fields are tolerated, which is how `openinfer: {...}` and
`reasoning_content` ride along.

Errors follow `{"error": {"message", "type", "code", "param"}}` with an HTTP status. The SDK
maps 404 to `NotFoundError`, 429 to `RateLimitError`, and so on. `openinfer/errors.py` produces
this shape from every exception type.

## Server-Sent Events

Streaming uses SSE: a `text/event-stream` response where each event is `data: <payload>\n\n`.
OpenAI streams one JSON chunk per event and ends with the literal `data: [DONE]`. Each chunk is
`object: "chat.completion.chunk"` with a `delta` instead of a `message`:

```
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}

data: [DONE]
```

Conventions OpenInfer follows (`openinfer/server/routes/chat.py`, `schemas.py`): the first
chunk carries `delta.role`, all chunks share one `id`, the final content chunk carries
`finish_reason`, a usage-only chunk with empty `choices` appears only when the client sent
`stream_options: {"include_usage": true}`, and `[DONE]` is always last. Headers set
`Cache-Control: no-cache` and `X-Accel-Buffering: no` so proxies like nginx do not buffer the
stream into one blob.

Why SSE and not WebSockets? SSE is plain HTTP: one request, one response that happens to be
long. It works through every proxy, needs no upgrade handshake, and the client is a `for` loop
over lines. WebSockets are bidirectional, which chat completion does not need. gRPC streaming
would be faster but no browser or SDK speaks it here.

## Two subtle bits

**Routing errors before the first byte.** Once a response has started with status 200 you
cannot change it to 404. The chat route primes the generator with `anext()` before building the
`StreamingResponse`; if the router raises (unknown model, privacy fail-closed) the client gets
a normal JSON error with the right status. Errors after the first chunk are sent as an
`{"error": ...}` event followed by `[DONE]`, which is what OpenAI does.

**Consuming SSE from a backend.** `iter_sse()` in `openai_compat.py` reads lines, ignores
comments and blank lines, strips the `data:` prefix, stops on `[DONE]` and decodes JSON.
Ollama's native API streams newline-delimited JSON instead (no `data:` prefix), handled in
`ollama.py`. Same idea, different framing.

## Where to look

- `openinfer/server/schemas.py`: request models and response builders.
- `openinfer/server/routes/chat.py`: the SSE generator.
- `tests/test_server_chat.py::test_chat_stream_sse`: what a correct stream looks like.

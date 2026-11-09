"""POST /v1/chat/completions, streaming and not."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from openinfer.errors import InvalidRequest, OpenInferError
from openinfer.ledger.base import LedgerEntry
from openinfer.router import Candidate, Router
from openinfer.server.schemas import (
    ChatCompletionRequest,
    chat_chunk,
    chat_completion_response,
    new_id,
    now,
    parse_hints,
    route_info,
    sse,
    usage_only_chunk,
)
from openinfer.types import ChatChunk, Usage

log = logging.getLogger("openinfer.server.chat")
router = APIRouter(prefix="/v1", tags=["chat"])

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # tell nginx style proxies not to buffer the stream
}


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    if not body.messages:
        raise InvalidRequest("messages must not be empty")
    if body.n not in (None, 1):
        raise InvalidRequest("n > 1 is not supported")
    r: Router = request.app.state.router
    hints = parse_hints(request.headers, body.openinfer, body.metadata)
    internal = body.to_internal(hints)
    completion_id = new_id("chatcmpl")
    created = now()

    if not body.stream:
        result, entry = await r.chat(internal)
        info = route_info(
            entry.backend,
            entry.model,
            alias=entry.alias,
            cost_usd=round(entry.cost_usd, 8),
            latency_ms=entry.latency_ms,
            attempts=entry.attempts,
            usage_exact=entry.usage_exact,
        )
        payload = chat_completion_response(
            result, completion_id=completion_id, created=created, info=info
        )
        return JSONResponse(
            payload,
            headers={
                "X-OpenInfer-Backend": entry.backend,
                "X-OpenInfer-Model": entry.model,
                "X-OpenInfer-Cost-USD": f"{entry.cost_usd:.8f}",
                "X-OpenInfer-Attempts": str(entry.attempts),
            },
        )

    # Streaming. Routing errors raised before the first chunk must still arrive as a normal
    # JSON error with a status code, so we prime the generator before building the response.
    include_usage = body.wants_stream_usage()
    gen = _stream(r, internal, completion_id, created, include_usage)
    try:
        first = await anext(gen)
    except StopAsyncIteration:
        first = sse("[DONE]")
    except OpenInferError as exc:
        return JSONResponse(exc.to_openai(), status_code=exc.status_code)

    async def body_iter() -> AsyncIterator[str]:
        yield first
        async for item in gen:
            yield item

    return StreamingResponse(body_iter(), media_type="text/event-stream", headers=STREAM_HEADERS)


async def _stream(
    r: Router,
    internal: Any,
    completion_id: str,
    created: int,
    include_usage: bool,
) -> AsyncIterator[str]:
    chosen: dict[str, Candidate] = {}
    done: dict[str, LedgerEntry] = {}

    def on_route(cand: Candidate) -> None:
        chosen["cand"] = cand

    def on_done(entry: LedgerEntry) -> None:
        done["entry"] = entry

    model_name = internal.model
    sent_role = False
    final_usage: Usage | None = None
    try:
        async for chunk in r.chat_stream(internal, on_route=on_route, on_done=on_done):
            cand = chosen.get("cand")
            if cand is not None:
                model_name = cand.model
            if not sent_role:
                sent_role = True
                info = route_info(cand.backend.name, cand.model) if cand else None
                yield sse(
                    chat_chunk(
                        completion_id=completion_id,
                        created=created,
                        model=model_name,
                        delta={"role": "assistant", "content": ""},
                        info=info,
                    )
                )
            for payload in _chunk_to_openai(chunk, completion_id, created, model_name):
                yield sse(payload)
            if chunk.usage is not None:
                final_usage = chunk.usage
    except OpenInferError as exc:
        if not sent_role:
            raise
        # Mid-stream failure: OpenAI style clients get an error event then the stream ends.
        yield sse({"error": exc.to_openai()["error"]})
        yield sse("[DONE]")
        return

    if include_usage:
        usage = final_usage
        entry = done.get("entry")
        if usage is None and entry is not None:
            usage = Usage(entry.prompt_tokens, entry.completion_tokens, exact=entry.usage_exact)
        if usage is not None:
            yield sse(
                usage_only_chunk(
                    completion_id=completion_id, created=created, model=model_name, usage=usage
                )
            )
    yield sse("[DONE]")


def _chunk_to_openai(
    chunk: ChatChunk, completion_id: str, created: int, model: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    delta: dict[str, Any] = {}
    if chunk.delta:
        delta["content"] = chunk.delta
    if chunk.reasoning:
        delta["reasoning_content"] = chunk.reasoning
    if chunk.tool_calls:
        delta["tool_calls"] = [
            {**tc, "index": tc.get("index", i)} for i, tc in enumerate(chunk.tool_calls)
        ]
    if delta or chunk.finish_reason:
        out.append(
            chat_chunk(
                completion_id=completion_id,
                created=created,
                model=model,
                delta=delta,
                finish_reason=chunk.finish_reason,
            )
        )
    return out

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from openinfer.errors import InvalidRequest
from openinfer.router import Router
from openinfer.server.schemas import EmbeddingsRequest, embeddings_response, parse_hints, route_info
from openinfer.types import EmbeddingRequest

router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embeddings")
async def create_embeddings(body: EmbeddingsRequest, request: Request) -> Any:
    inputs = body.inputs()
    if not inputs:
        raise InvalidRequest("input must not be empty")
    if body.encoding_format not in (None, "float", "base64"):
        raise InvalidRequest("encoding_format must be 'float' or 'base64'")
    r: Router = request.app.state.router
    hints = parse_hints(request.headers, body.openinfer, body.metadata)
    result, entry = await r.embed(
        EmbeddingRequest(model=body.model, inputs=inputs, dimensions=body.dimensions, hints=hints)
    )
    info = route_info(
        entry.backend,
        entry.model,
        alias=entry.alias,
        cost_usd=round(entry.cost_usd, 8),
        latency_ms=entry.latency_ms,
        attempts=entry.attempts,
    )
    return JSONResponse(
        embeddings_response(result, info, base64_encode=body.encoding_format == "base64"),
        headers={"X-OpenInfer-Backend": entry.backend, "X-OpenInfer-Model": entry.model},
    )

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from openinfer.errors import ModelNotFound
from openinfer.router import Router

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    r: Router = request.app.state.router
    return {"object": "list", "data": r.model_cards()}


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> dict[str, Any]:
    r: Router = request.app.state.router
    for card in r.model_cards():
        if card["id"] == model_id:
            return card
    raise ModelNotFound(f"model {model_id!r} not found")

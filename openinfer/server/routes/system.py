"""Operational routes: /health, /v1/hardware, /v1/ledger, /metrics, and a root redirect."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import RedirectResponse

from openinfer import __version__
from openinfer.errors import NotSupported
from openinfer.hardware import detect
from openinfer.router import Router

router = APIRouter(tags=["system"])


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@router.get("/health")
async def health(request: Request, deep: bool = Query(default=True)) -> dict[str, Any]:
    """Liveness plus, by default, a quick probe of every backend (3s cap each, in parallel)."""
    r: Router = request.app.state.router
    cfg = request.app.state.config
    payload: dict[str, Any] = {
        "status": "ok",
        "version": __version__,
        "config": cfg.source,
        "aliases": list(cfg.aliases),
    }
    if deep:
        statuses = await r.health()
        payload["backends"] = {
            name: {
                "ok": status.ok,
                "detail": status.detail,
                "local": r.backends[name].local,
                "type": r.backends[name].kind,
            }
            for name, status in statuses.items()
        }
        payload["status"] = "ok" if any(s.ok for s in statuses.values()) else "degraded"
    return payload


@router.get("/v1/hardware")
async def hardware() -> dict[str, Any]:
    # Detection shells out to nvidia-smi etc.; keep the event loop free.
    return await asyncio.to_thread(detect)


@router.get("/v1/ledger")
async def ledger(
    request: Request,
    hours: float = Query(default=24.0, gt=0, le=24 * 365),
    limit: int = Query(default=20, ge=0, le=500),
) -> dict[str, Any]:
    led = request.app.state.ledger
    stats = await asyncio.to_thread(led.stats, hours)
    stats["recent"] = await asyncio.to_thread(led.recent, limit) if limit else []
    return stats


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    m = request.app.state.metrics
    if not m.enabled:
        raise NotSupported("prometheus_client is not installed: uv sync --extra metrics")
    body, content_type = m.render()
    return Response(content=body, media_type=content_type)

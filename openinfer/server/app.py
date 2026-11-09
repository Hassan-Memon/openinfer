"""FastAPI application factory.

`create_app()` wires config -> backends -> ledger -> router and hangs them on `app.state`
so route handlers stay thin. Tests call `create_app(config=..., backends=...)` with fakes.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from openinfer import __version__
from openinfer.backends import build_backends
from openinfer.backends.base import Backend
from openinfer.config import AppConfig, load_config
from openinfer.errors import AuthError, OpenInferError
from openinfer.ledger import Ledger, open_ledger
from openinfer.metrics import Metrics
from openinfer.pricing import Pricing
from openinfer.router import Router
from openinfer.server.routes import audio, chat, embeddings, models, system

log = logging.getLogger("openinfer.server")

OPEN_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc", "/metrics"}


def create_app(
    config: AppConfig | None = None,
    *,
    backends: dict[str, Backend] | None = None,
    ledger: Ledger | None = None,
    metrics: Metrics | None = None,
    pricing: Pricing | None = None,
    router: Router | None = None,
) -> FastAPI:
    cfg = config or load_config()
    backends = backends if backends is not None else build_backends(cfg)
    ledger = ledger or open_ledger(cfg.ledger.url)
    metrics = metrics or Metrics()
    router = router or Router(cfg, backends, ledger, pricing, metrics=metrics)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "OpenInfer %s: config=%s backends=%s aliases=%s",
            __version__,
            cfg.source,
            ",".join(backends) or "none",
            ",".join(cfg.aliases) or "none",
        )
        try:
            yield
        finally:
            await router.aclose()
            ledger.close()

    app = FastAPI(
        title="OpenInfer",
        version=__version__,
        summary="OpenAI-compatible inference runtime with privacy, cost and latency routing.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.config = cfg
    app.state.backends = backends
    app.state.ledger = ledger
    app.state.metrics = metrics
    app.state.router = router

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def bearer_auth(request: Request, call_next: Any) -> Any:
        key = cfg.server.api_key
        if key and request.url.path not in OPEN_PATHS and request.method != "OPTIONS":
            header = request.headers.get("authorization", "")
            token = header[7:] if header.lower().startswith("bearer ") else ""
            if token != key:
                err = AuthError("missing or invalid bearer token")
                return JSONResponse(err.to_openai(), status_code=err.status_code)
        return await call_next(request)

    @app.exception_handler(OpenInferError)
    async def handle_openinfer_error(_: Request, exc: OpenInferError) -> JSONResponse:
        return JSONResponse(exc.to_openai(), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "message": "invalid request body: "
                    + "; ".join(
                        f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg')}"
                        for e in exc.errors()
                    ),
                    "type": "invalid_request_error",
                    "code": "validation_error",
                    "param": None,
                }
            },
            status_code=400,
        )

    app.include_router(system.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(embeddings.router)
    app.include_router(audio.router)
    return app

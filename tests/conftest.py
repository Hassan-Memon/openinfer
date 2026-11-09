from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from openinfer.config import AppConfig, parse_config
from openinfer.ledger.sqlite import SQLiteLedger
from openinfer.metrics import Metrics
from openinfer.pricing import Pricing
from openinfer.router import Router
from openinfer.server import create_app

ECHO_YAML = """
server:
  api_key: null
ledger:
  url: "sqlite:///:memory:"
backends:
  echo:
    type: echo
aliases:
  fast: [echo/echo]
  smart: [echo/echo]
  private: [echo/echo]
  embed: [echo/echo-embed]
policy:
  retries: 1
  backoff_base_ms: 1
  backoff_max_ms: 2
"""


def make_config(text: str = ECHO_YAML, env: dict[str, str] | None = None) -> AppConfig:
    return parse_config(text, source="<test>", env=env or {})


@pytest.fixture
def echo_config() -> AppConfig:
    return make_config()


@pytest.fixture
def app(echo_config: AppConfig):
    return create_app(echo_config, metrics=Metrics(enabled=True))


@pytest.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


class NoSleep:
    """Replacement for asyncio.sleep that records requested delays and returns at once."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def make_router(
    config: AppConfig,
    backends: dict,
    *,
    pricing: Pricing | None = None,
    ledger: SQLiteLedger | None = None,
) -> tuple[Router, NoSleep, SQLiteLedger]:
    sleeper = NoSleep()
    led = ledger or SQLiteLedger("sqlite:///:memory:")
    router = Router(
        config,
        backends,
        led,
        pricing or Pricing.load_default(),
        metrics=Metrics(enabled=False),
        sleep=sleeper,
    )
    return router, sleeper, led

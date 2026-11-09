from __future__ import annotations

import pytest

from openinfer.config import ModelPrice
from openinfer.pricing import Pricing


def test_default_table_loads_and_prices_known_models():
    p = Pricing.load_default()
    assert {"groq", "openai", "gemini", "openrouter"} <= set(p.providers())
    price = p.price(backend="groq", provider="groq", model="llama-3.1-8b-instant", local=False)
    assert price.known and price.input_per_1m == 0.05 and price.output_per_1m == 0.08
    assert price.cost(1_000_000, 1_000_000) == pytest.approx(0.13)


def test_unknown_cloud_model_is_zero_but_flagged():
    p = Pricing.load_default()
    price = p.price(backend="x", provider="groq", model="mystery", local=False)
    assert price.known is False
    assert price.cost(10_000, 10_000) == 0.0


def test_local_is_free_unless_overridden():
    p = Pricing.load_default()
    assert p.price(backend="ollama", provider=None, model="llama3", local=True).cost(1, 1) == 0
    p.add_overrides("ollama", {"llama3": ModelPrice(input=0.1, output=0.2)})
    price = p.price(backend="ollama", provider=None, model="llama3", local=True)
    assert price.known and price.cost(1_000_000, 0) == pytest.approx(0.1)


def test_model_found_under_another_provider():
    p = Pricing.load_default()
    # OpenRouter style id looked up without a provider hint
    price = p.price(
        backend="custom", provider=None, model="meta-llama/llama-3.1-8b-instruct", local=False
    )
    assert price.known and price.output_per_1m == 0.05

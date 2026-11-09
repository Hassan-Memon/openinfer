"""Cost accounting: look up per token prices and turn token counts into dollars.

See docs/learn/03-concepts/cost-accounting-for-llms.md for why prices are per million tokens,
why input and output are priced separately, and why local backends are not free in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import yaml

from openinfer.config import ModelPrice

MILLION = 1_000_000


@dataclass(slots=True, frozen=True)
class Price:
    input_per_1m: float
    output_per_1m: float
    known: bool  # False when we have no entry and assume zero

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_1m + completion_tokens * self.output_per_1m
        ) / MILLION


ZERO = Price(0.0, 0.0, known=True)
UNKNOWN = Price(0.0, 0.0, known=False)


class Pricing:
    def __init__(self, table: dict[str, dict[str, ModelPrice]] | None = None) -> None:
        self._table: dict[str, dict[str, ModelPrice]] = table or {}
        # backend name -> model -> price overrides from openinfer.yaml
        self._overrides: dict[str, dict[str, ModelPrice]] = {}

    @classmethod
    def load_default(cls) -> Pricing:
        text = resources.files("openinfer").joinpath("pricing.yaml").read_text("utf-8")
        raw = yaml.safe_load(text) or {}
        table: dict[str, dict[str, ModelPrice]] = {}
        for provider, models in (raw.get("providers") or {}).items():
            table[provider] = {
                model: ModelPrice.model_validate(price) for model, price in (models or {}).items()
            }
        return cls(table)

    def add_overrides(self, backend: str, prices: dict[str, ModelPrice]) -> None:
        if prices:
            self._overrides.setdefault(backend, {}).update(prices)

    def price(self, *, backend: str, provider: str | None, model: str, local: bool) -> Price:
        override = self._overrides.get(backend, {}).get(model)
        if override is not None:
            return Price(override.input, override.output, known=True)
        if local:
            return ZERO
        if provider and model in self._table.get(provider, {}):
            entry = self._table[provider][model]
            return Price(entry.input, entry.output, known=True)
        # Same model name under any provider (OpenRouter style ids, vendor renames).
        for models in self._table.values():
            if model in models:
                entry = models[model]
                return Price(entry.input, entry.output, known=True)
        return UNKNOWN

    def providers(self) -> list[str]:
        return sorted(self._table)

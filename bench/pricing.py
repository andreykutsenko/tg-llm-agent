"""Price list per model, USD per million tokens — the only place with these numbers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # llms импортирует прайс, поэтому обратный импорт только для типов
    from llms.protocol import Usage

TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cached_input: float
    cache_write: float


# Тарифы Anthropic API на 2026-09: чтение из кэша ×0.1, запись в кэш ×1.25.
PRICES: dict[str, Price] = {
    "claude-opus-5": Price(input=5.0, output=25.0, cached_input=0.5, cache_write=6.25),
    "claude-sonnet-5": Price(input=2.0, output=10.0, cached_input=0.2, cache_write=2.5),
    "claude-haiku-4-5": Price(input=1.0, output=5.0, cached_input=0.1, cache_write=1.25),
}
# Локальная модель ничего не стоит; неизвестная облачная — тоже 0, но с пометкой.
FREE = Price(input=0.0, output=0.0, cached_input=0.0, cache_write=0.0)


def price_for(model: str) -> Price:
    return PRICES.get(model, FREE)


@dataclass(frozen=True)
class CostBreakdown:
    input: float
    output: float
    cached_input: float
    cache_write: float

    @property
    def total(self) -> float:
        return self.input + self.output + self.cached_input + self.cache_write


def cost_breakdown(model: str, usage: "Usage") -> CostBreakdown:
    price = price_for(model)
    unit = TOKENS_PER_PRICE_UNIT
    return CostBreakdown(
        input=usage.input_tokens * price.input / unit,
        output=usage.output_tokens * price.output / unit,
        cached_input=usage.cached_input_tokens * price.cached_input / unit,
        cache_write=usage.cache_write_input_tokens * price.cache_write / unit,
    )


def estimate_cost(model: str, usage: "Usage") -> float:
    return cost_breakdown(model, usage).total

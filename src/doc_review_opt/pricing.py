"""Token prices and cost estimates (Standard tier, verified 2026-11-18)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from doc_review_opt.config import get_settings
from doc_review_opt.models import TokenUsage

SETTINGS = get_settings()
BATCH_DISCOUNT = SETTINGS.pricing.batch_discount
CACHE_MIN_TOKENS = SETTINGS.pricing.cache_min_tokens
DECISIONS_PRICE_KEY = SETTINGS.models.decisions_price_key
_PER_MILLION = 1_000_000

MODEL_PRICES_USD_PER_1M: Mapping[str, Mapping[str, float]] = {
    model: row.model_dump() for model, row in SETTINGS.prices.items()
}


class UnknownModelError(ValueError): ...


def approx_tokens(value: Any) -> int:
    if value is None:
        return 0
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return max(1, math.ceil(len(text) / 4))


def estimate_cost_usd(model: str, usage: TokenUsage, *, discount: float = 1.0) -> float:
    rates = MODEL_PRICES_USD_PER_1M.get(model)
    if rates is None:
        raise UnknownModelError(f"no price for model {model!r}")
    cached = min(usage.cached, usage.input)
    written = min(usage.cache_write, usage.input - cached)
    plain = usage.input - cached - written
    cost = (
        plain * rates["input"]
        + cached * rates["cached_input"]
        + written * rates["cache_write"]
        + usage.output * rates["output"]
    ) / _PER_MILLION
    return cost * discount

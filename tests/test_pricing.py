"""Tests for token pricing."""

import pytest

from doc_review_opt.models import TokenUsage
from doc_review_opt.pricing import (
    BATCH_DISCOUNT,
    DECISIONS_PRICE_KEY,
    UnknownModelError,
    approx_tokens,
    estimate_cost_usd,
)


def test_plain_input_and_output() -> None:
    usage = TokenUsage(input=1_000_000, output=1_000_000)
    assert estimate_cost_usd("gpt-6.1-terra", usage) == pytest.approx(6.00)


def test_cached_tokens_are_a_subset_of_input() -> None:
    usage = TokenUsage(input=1_000_000, cached=400_000)
    expected = 0.6 * 1.00 + 0.4 * 0.05
    assert estimate_cost_usd("gpt-6.1-terra", usage) == pytest.approx(expected)


def test_cache_writes_use_the_write_rate() -> None:
    usage = TokenUsage(input=1_000_000, cache_write=1_000_000)
    assert estimate_cost_usd("gpt-6.1-terra", usage) == pytest.approx(1.25)
    assert estimate_cost_usd("gpt-6-luna", usage) == pytest.approx(0.125)


def test_reasoning_tokens_are_not_billed_twice() -> None:
    plain = TokenUsage(output=1_000)
    with_reasoning = TokenUsage(output=1_000, reasoning=900)
    assert estimate_cost_usd("gpt-6-luna", plain) == estimate_cost_usd(
        "gpt-6-luna", with_reasoning
    )


def test_batch_discount_halves_the_cost() -> None:
    usage = TokenUsage(input=10_000, output=2_000)
    full = estimate_cost_usd("gpt-6-luna", usage)
    batch = estimate_cost_usd("gpt-6-luna", usage, discount=BATCH_DISCOUNT)
    assert batch == pytest.approx(full / 2)


def test_decisions_output_tokens_are_free() -> None:
    usage = TokenUsage(input=1_000_000, output=5_000_000)
    assert estimate_cost_usd(DECISIONS_PRICE_KEY, usage) == pytest.approx(0.10)


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(UnknownModelError):
        estimate_cost_usd("gpt-unpriced-model", TokenUsage(input=1))


def test_usage_adds_fieldwise() -> None:
    total = TokenUsage(input=1, cached=1, output=2) + TokenUsage(input=3, reasoning=1)
    assert total == TokenUsage(input=4, cached=1, output=2, reasoning=1)


def test_approx_tokens() -> None:
    assert approx_tokens(None) == 0
    assert approx_tokens("a" * 8) == 2
    assert approx_tokens({"a": "b"}) == 3

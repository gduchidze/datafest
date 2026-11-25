"""Tests for the packaged config.toml loader."""

import copy
import tomllib
from importlib.resources import files
from typing import Any

import pytest

from doc_review_opt.agent.routing import DEFAULT_POLICY
from doc_review_opt.agent.variants import CALIBRATED_POLICY, VARIANT_ORDER
from doc_review_opt.config import get_settings, parse_settings
from doc_review_opt.exceptions import ConfigError
from doc_review_opt.pricing import BATCH_DISCOUNT, MODEL_PRICES_USD_PER_1M


def raw() -> dict[str, Any]:
    text = files("doc_review_opt").joinpath("config.toml").read_text("utf-8")
    return tomllib.loads(text)


def test_packaged_toml_loads_with_every_section() -> None:
    settings = get_settings()
    assert settings.live.budget_limit_usd == 1.0
    assert settings.live.reasoning_effort == "none"
    assert set(VARIANT_ORDER) <= set(settings.variants)
    assert len(settings.scenarios) == 6


def test_models_are_limited_to_terra_and_luna() -> None:
    models = get_settings().models
    allowed = {"gpt-6.1-terra", "gpt-6-luna"}
    assert {models.decisions, models.terra, models.luna, models.judge} <= allowed
    assert {models.triage, models.redaction, models.second_opinion} <= allowed


def test_policy_and_pricing_come_from_the_toml() -> None:
    assert DEFAULT_POLICY.privilege_escalation == 0.2
    assert DEFAULT_POLICY.responsive_confidence == 0.7
    assert (
        CALIBRATED_POLICY.responsive_confidence < DEFAULT_POLICY.responsive_confidence
    )
    assert BATCH_DISCOUNT == 0.5
    assert MODEL_PRICES_USD_PER_1M["decisions:gpt-6-luna"]["output"] == 0.0


def test_missing_key_fails_with_a_clear_error() -> None:
    broken = copy.deepcopy(raw())
    del broken["live"]["budget_limit_usd"]
    with pytest.raises(ConfigError, match=r"live\.budget_limit_usd"):
        parse_settings(broken)


def test_unknown_key_is_rejected() -> None:
    broken = copy.deepcopy(raw())
    broken["live"]["surprise"] = 1
    with pytest.raises(ConfigError, match="surprise"):
        parse_settings(broken)


def test_toml_is_inside_the_package_directory() -> None:
    assert files("doc_review_opt").joinpath("config.toml").is_file()


def test_calibration_grid_searches_the_privilege_threshold() -> None:
    grid = get_settings().calibration.privilege_escalation
    assert tuple(grid) == (0.2, 0.25, 0.3, 0.35, 0.4, 0.45)


def test_palette_and_live_run_load() -> None:
    settings = get_settings()
    assert settings.palette.surface == "#fcfcfb"
    assert settings.palette.blue == "#2a78d6"
    assert settings.palette.critical == "#d03b3b"
    assert set(settings.entity_colors.values()) <= set(settings.palette.model_dump())
    assert settings.live_run.pages == settings.live.page_limit
    assert [r.variant for r in settings.live_run.rounds] == VARIANT_ORDER

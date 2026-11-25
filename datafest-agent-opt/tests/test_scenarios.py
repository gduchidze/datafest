"""Tests for the scenario fit score and best fit."""

import asyncio
from typing import Any

import pandas as pd
import pytest

from doc_review_opt.experiment.scenarios import (
    ARCHITECTURE_OPTIONS,
    OPERATING_SCENARIOS,
    architecture_metrics,
    best_fit,
    scenario_fit_score,
    scenario_map,
)
from doc_review_opt.experiment.simulation import simulate_all
from doc_review_opt.experiment.traces import traces_frame
from doc_review_opt.reporting.reporting import summarize


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return summarize(traces_frame(asyncio.run(simulate_all(include_extras=True))))


def fake_summary(**overrides: float) -> pd.DataFrame:
    row: dict[str, Any] = {
        "mean_quality": 0.95,
        "policy_compliance": 1.0,
        "escalation_rate": 0.30,
        "p50_latency_s": 1.0,
        "model_cost_per_page_usd": 0.0005,
        "monthly_model_cost_at_100k_pages": 50.0,
        "background_cost_per_page_usd": 0.0001,
    }
    variants = {o["variant"] for o in ARCHITECTURE_OPTIONS.values() if o["variant"]}
    return pd.DataFrame([{**row, **overrides, "variant": v} for v in variants])


SCENARIO = {
    "scenario": "S",
    "description": "d",
    "quality_floor": 0.9,
    "policy_floor": 0.95,
    "p50_latency_target_s": 2.0,
    "monthly_budget_100k_pages_usd": 100,
    "max_escalation_rate": 0.4,
    "needs_async": True,
    "cache_locality": "low",
    "recommended": ["controlled_terra"],
}


def test_options_and_scenarios_match_the_brief() -> None:
    assert len(OPERATING_SCENARIOS) == 6
    assert {
        "controlled_terra",
        "llm_routed",
        "decisions_code_router",
        "decisions_llm_cascade",
        "decisions_cache",
        "balanced_split",
        "decisions_split_no_cache",
        "baseline_one_big_agent",
    } == set(ARCHITECTURE_OPTIONS)
    assert {
        "quality_floor",
        "policy_floor",
        "p50_latency_target_s",
        "monthly_budget_100k_pages_usd",
        "needs_async",
        "cache_locality",
        "recommended",
        "max_escalation_rate",
    } <= set(OPERATING_SCENARIOS[0])


def test_all_constraints_pass_scores_the_cookbook_points() -> None:
    result = scenario_fit_score(SCENARIO, "controlled_terra", fake_summary())
    assert result["score"] == 3 + 3 + 2 + 2 + 2 - 1 + 1
    assert result["failed_constraints"] == "async split"


def test_every_failure_and_the_cache_penalty() -> None:
    bad = fake_summary(
        mean_quality=0.5,
        policy_compliance=0.5,
        p50_latency_s=9.0,
        monthly_model_cost_at_100k_pages=1e6,
        escalation_rate=0.9,
    )
    result = scenario_fit_score(SCENARIO, "balanced_split", bad)
    assert result["score"] == -4 - 5 - 2 - 2 + 2 - 2 - 3
    assert result["failed_constraints"] == (
        "quality, policy, latency, budget, escalation rate, cache locality"
    )
    medium = {**SCENARIO, "cache_locality": "medium", "recommended": ["balanced_split"]}
    assert (
        scenario_fit_score(medium, "balanced_split", bad)["score"]
        == -4 - 5 - 2 - 2 + 2 - 2 - 1 + 1
    )


def test_synthetic_option_is_computed_from_two_variants(summary: pd.DataFrame) -> None:
    metrics = architecture_metrics("decisions_split_no_cache", summary)
    base = summary[summary["variant"] == "03_decision_model"].iloc[0]
    split = summary[summary["variant"] == "05_split_workflow"].iloc[0]
    assert metrics["p50_latency_s"] == pytest.approx(
        max(base["p50_latency_s"] - 0.30, 0.5)
    )
    expected = (
        max(base["model_cost_per_page_usd"] - 0.00007, 0)
        + split["background_cost_per_page_usd"]
    )
    assert metrics["cost_per_page_usd"] == pytest.approx(expected)
    assert metrics["monthly_cost_at_100k_pages"] == pytest.approx(expected * 100_000)


def test_best_fit_picks_one_row_per_scenario_highest_score_first(
    summary: pd.DataFrame,
) -> None:
    mapped = scenario_map(summary)
    assert len(mapped) == len(OPERATING_SCENARIOS) * len(ARCHITECTURE_OPTIONS)
    best = best_fit(summary)
    assert list(best["scenario"]) == [s["scenario"] for s in OPERATING_SCENARIOS]
    for _, row in best.iterrows():
        group = mapped[mapped["scenario"] == row["scenario"]]
        assert row["score"] == group["score"].max()
        top = group[group["score"] == group["score"].max()]
        assert (
            row["monthly_cost_at_100k_pages"] == top["monthly_cost_at_100k_pages"].min()
        )
    assert "baseline_one_big_agent" not in set(best["architecture"])

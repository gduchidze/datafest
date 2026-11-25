"""Architecture options mapped to operating scenarios with a fit score."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from doc_review_opt.config import get_settings

SETTINGS = get_settings()
POINTS = SETTINGS.scoring
SYNTHETIC = SETTINGS.synthetic
SYNTHETIC_VARIANT = ""
ARCHITECTURE_OPTIONS: dict[str, dict[str, Any]] = {
    key: option.model_dump() for key, option in SETTINGS.architecture.items()
}
OPERATING_SCENARIOS: list[dict[str, Any]] = [
    scenario.model_dump() for scenario in SETTINGS.scenarios
]


def _row(summary: pd.DataFrame, variant: str) -> dict[str, Any]:
    row = summary.loc[summary["variant"] == variant].iloc[0]
    return {str(key): value for key, value in row.to_dict().items()}


def _metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "quality": row["mean_quality"],
        "policy_compliance": row["policy_compliance"],
        "escalation_rate": row["escalation_rate"],
        "p50_latency_s": row["p50_latency_s"],
        "cost_per_page_usd": row["model_cost_per_page_usd"],
        "monthly_cost_at_100k_pages": row["monthly_model_cost_at_100k_pages"],
    }


def architecture_metrics(option_key: str, summary: pd.DataFrame) -> dict[str, Any]:
    option = ARCHITECTURE_OPTIONS[option_key]
    if option["variant"] != SYNTHETIC_VARIANT:
        return _metrics(_row(summary, option["variant"]))
    base = _row(summary, SYNTHETIC.base_variant)
    split = _row(summary, SYNTHETIC.split_variant)
    cost = (
        max(base["model_cost_per_page_usd"] - SYNTHETIC.sync_background_cost_usd, 0)
        + split["background_cost_per_page_usd"]
    )
    return {
        **_metrics(base),
        "p50_latency_s": max(
            base["p50_latency_s"] - SYNTHETIC.latency_saving_s,
            SYNTHETIC.latency_floor_s,
        ),
        "cost_per_page_usd": cost,
        "monthly_cost_at_100k_pages": cost * POINTS.monthly_pages,
    }


def _cache_penalty(scenario: Mapping[str, Any], option: Mapping[str, Any]) -> int:
    if option["cache"] == "none":
        return 0
    penalties = {"low": POINTS.cache_penalty_low, "medium": POINTS.cache_penalty_medium}
    return penalties.get(scenario["cache_locality"], 0)


def scenario_fit_score(
    scenario: Mapping[str, Any], option_key: str, summary: pd.DataFrame
) -> dict[str, Any]:
    option = ARCHITECTURE_OPTIONS[option_key]
    metrics = architecture_metrics(option_key, summary)
    checks = {
        "quality": (
            metrics["quality"] >= scenario["quality_floor"],
            POINTS.quality_ok,
            POINTS.quality_bad,
        ),
        "policy": (
            metrics["policy_compliance"] >= scenario["policy_floor"],
            POINTS.policy_ok,
            POINTS.policy_bad,
        ),
        "latency": (
            metrics["p50_latency_s"] <= scenario["p50_latency_target_s"],
            POINTS.latency_ok,
            POINTS.latency_bad,
        ),
        "budget": (
            metrics["monthly_cost_at_100k_pages"]
            <= scenario["monthly_budget_100k_pages_usd"],
            POINTS.budget_ok,
            POINTS.budget_bad,
        ),
        "async split": (
            not scenario["needs_async"] or "async" in option["workflow"],
            POINTS.async_ok,
            POINTS.async_bad,
        ),
        "escalation rate": (
            metrics["escalation_rate"] <= scenario["max_escalation_rate"],
            POINTS.escalation_ok,
            POINTS.escalation_bad,
        ),
    }
    penalty = _cache_penalty(scenario, option)
    score = sum(ok if passed else bad for passed, ok, bad in checks.values()) - penalty
    if option_key in scenario["recommended"]:
        score += POINTS.recommended_bonus
    failed = [name for name, (passed, _, _) in checks.items() if not passed]
    if penalty:
        failed.append("cache locality")
    return {
        "scenario": scenario["scenario"],
        "architecture": option_key,
        "label": option["label"],
        "score": score,
        **metrics,
        "failed_constraints": ", ".join(failed) or "none",
    }


def scenario_map(
    summary: pd.DataFrame, scenarios: Sequence[Mapping[str, Any]] = OPERATING_SCENARIOS
) -> pd.DataFrame:
    rows = [
        scenario_fit_score(scenario, key, summary)
        for scenario in scenarios
        for key in ARCHITECTURE_OPTIONS
    ]
    return pd.DataFrame(rows)


def best_fit(summary: pd.DataFrame) -> pd.DataFrame:
    scored = scenario_map(summary)
    order = {s["scenario"]: i for i, s in enumerate(OPERATING_SCENARIOS)}
    scored["_order"] = scored["scenario"].map(order)
    ranked = scored.sort_values(
        ["_order", "score", "monthly_cost_at_100k_pages"], ascending=[True, False, True]
    )
    return ranked.groupby("scenario", sort=False).head(1).drop(columns="_order")

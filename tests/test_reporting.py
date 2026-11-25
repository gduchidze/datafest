"""Tests for summaries, round impact and charts."""

import asyncio

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from doc_review_opt.agent.variants import BASELINE, CONTROLS, VARIANT_ORDER
from doc_review_opt.config import get_settings
from doc_review_opt.experiment.simulation import simulate_all
from doc_review_opt.experiment.traces import traces_frame
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.reporting.chart_style import PALETTE, color, signed_percent
from doc_review_opt.reporting.reporting import (
    per_page_view,
    plot_cost_latency,
    plot_cost_split,
    plot_eval_set,
    plot_progress,
    plot_quality_gate,
    plot_round_chart,
    plot_route_mix,
    plot_scorecard,
    plot_sim_vs_live,
    recorded_live_note,
    recorded_live_summary,
    round_impact,
    summarize,
)
from tests.helpers import assert_clean_chart, at, bar_colors


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return traces_frame(asyncio.run(simulate_all()))


@pytest.fixture(scope="module")
def summary(frame: pd.DataFrame) -> pd.DataFrame:
    return summarize(frame)


def test_summary_has_the_cookbook_columns(summary: pd.DataFrame) -> None:
    expected = {
        "pages",
        "mean_quality",
        "policy_compliance",
        "route_accuracy",
        "escalation_accuracy",
        "escalation_rate",
        "privileged_leaks",
        "pii_leaks",
        "mean_tool_calls",
        "mean_extra_tool_calls",
        "mean_input_tokens",
        "mean_cached_tokens",
        "mean_output_tokens",
        "mean_sync_tokens",
        "mean_total_tokens",
        "p50_latency_s",
        "p95_latency_s",
        "sync_cost_per_page_usd",
        "background_cost_per_page_usd",
        "model_cost_per_page_usd",
        "attorney_cost_per_page_usd",
        "cost_per_page_usd",
        "blended_cost_per_verified_page",
        "cost_reduction_vs_baseline",
        "token_reduction_vs_baseline",
        "latency_reduction_vs_baseline",
        "monthly_cost_at_100k_pages",
    }
    assert expected <= set(summary.columns)
    assert list(summary["variant"]) == VARIANT_ORDER
    assert at(summary, 0, "cost_reduction_vs_baseline") == 0


def test_blended_cost_per_verified_page_divides_by_correct_pages(
    summary: pd.DataFrame,
) -> None:
    row = summary.iloc[0]
    assert row["blended_cost_per_verified_page"] == pytest.approx(
        row["total_cost_usd"] / row["correct_pages"]
    )
    assert row["monthly_cost_at_100k_pages"] == pytest.approx(
        row["cost_per_page_usd"] * 100_000
    )


def test_round_impact_is_the_diff_from_the_previous_round(
    summary: pd.DataFrame,
) -> None:
    impact = round_impact(summary)
    assert pd.isna(at(impact, 0, "mean_sync_tokens_delta_vs_previous"))
    assert at(impact, 1, "mean_sync_tokens_delta_vs_previous") == pytest.approx(
        at(summary, 1, "mean_sync_tokens") - at(summary, 0, "mean_sync_tokens")
    )
    assert at(impact, 1, "quality_delta_vs_previous") > 0


def test_per_page_view_filters_variants_and_columns(frame: pd.DataFrame) -> None:
    view = per_page_view(frame, [BASELINE, CONTROLS], ["route", "cost_usd"])
    assert list(view.columns) == ["doc_id", "variant_label", "route", "cost_usd"]
    assert set(view["variant_label"]) == {"Bad baseline", "Round 1: controls"}


def test_charts_return_figures_with_short_labels(summary: pd.DataFrame) -> None:
    fig = plot_round_chart(summary)
    assert_clean_chart(fig, 3)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "R1: controls" in labels
    progress = plot_progress(summary, CONTROLS)
    assert_clean_chart(progress, 3)
    assert len(progress.axes[0].get_xticklabels()) == 2


@pytest.fixture(scope="module")
def eval_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": page.doc_id,
            "difficulty": page.labels.difficulty,
            "expected_route": page.labels.expected_route,
        }
        for page in EVAL_SET
    )


def test_round_charts_use_palette_colors(summary: pd.DataFrame) -> None:
    fig = plot_round_chart(summary)
    assert bar_colors(fig.axes[0])[0] == color("sync_tokens")
    assert bar_colors(fig.axes[1])[0] == PALETTE.blue


def test_eval_set_chart_colors_follow_routes(eval_table: pd.DataFrame) -> None:
    fig = plot_eval_set(eval_table)
    assert_clean_chart(fig, 2)
    assert bar_colors(fig.axes[0])[:2] == [PALETTE.blue, PALETTE.orange]
    assert len(fig.axes[1].patches) == eval_table["difficulty"].nunique()


def test_cost_charts_label_bars_and_stack_costs(summary: pd.DataFrame) -> None:
    fig = plot_cost_latency(summary)
    assert_clean_chart(fig, 2)
    assert bar_colors(fig.axes[0])[0] == PALETTE.blue
    assert fig.axes[0].get_yscale() == "log"
    assert len(fig.axes[0].texts) == len(summary)
    split = plot_cost_split(summary)
    assert_clean_chart(split, 1)
    assert bar_colors(split.axes[0])[-1] == PALETTE.orange


def test_quality_gate_chart_pairs_color_with_text() -> None:
    gates = pd.DataFrame(
        {
            "label": ["a", "b"],
            "pass_rate": [0.99, 0.5],
            "accepted": [True, False],
        }
    )
    fig = plot_quality_gate(gates, 0.95)
    assert_clean_chart(fig, 1)
    assert bar_colors(fig.axes[0]) == [PALETTE.good, PALETTE.critical]
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("✓ ACCEPTED" in t for t in texts)
    assert any("✗ REJECTED" in t for t in texts)


def test_route_mix_and_scorecard_cover_every_round(frame: pd.DataFrame) -> None:
    mix = plot_route_mix(frame, VARIANT_ORDER)
    assert_clean_chart(mix, 1)
    assert bar_colors(mix.axes[0])[0] == PALETTE.blue
    assert len(mix.axes[0].get_yticklabels()) == len(VARIANT_ORDER) + 1
    assert mix.axes[0].get_xlabel() == f"share of the {len(EVAL_SET)} pages"
    card = plot_scorecard(frame, VARIANT_ORDER)
    assert_clean_chart(card, 1)
    assert len(card.axes[0].get_yticklabels()) == len(EVAL_SET)
    points = sum(len(np.asarray(c.get_offsets())) for c in card.axes[0].collections)
    assert points == len(EVAL_SET) * len(VARIANT_ORDER)


def test_recorded_live_run_comes_from_the_config() -> None:
    live = get_settings().live_run
    recorded = recorded_live_summary()
    assert list(recorded["variant"]) == [row.variant for row in live.rounds]
    assert "6 pages" in recorded_live_note()
    assert live.date in recorded_live_note()


def test_sim_vs_live_uses_blue_and_orange(summary: pd.DataFrame) -> None:
    fig = plot_sim_vs_live(summary, recorded_live_summary(), recorded_live_note())
    assert_clean_chart(fig, 2)
    colors = bar_colors(fig.axes[0])
    assert colors[0] == PALETTE.blue
    assert colors[-1] == PALETTE.orange
    assert recorded_live_note() in fig.get_suptitle()


def test_sim_vs_live_tolerates_a_partial_live_run(summary: pd.DataFrame) -> None:
    partial = recorded_live_summary().head(2)
    fig = plot_sim_vs_live(summary, partial, "live: 2 pages")
    assert_clean_chart(fig, 2)


def test_signed_percent_never_prints_a_negative_zero() -> None:
    assert signed_percent(-0.0004) == "0%"
    assert signed_percent(0.0) == "0%"
    assert signed_percent(0.126) == "+13%"
    assert signed_percent(-0.5) == "-50%"


def test_log_cost_charts_use_snug_y_limits(summary: pd.DataFrame) -> None:
    costs = summary["model_cost_per_page_usd"]
    live = recorded_live_summary()
    figs = [
        plot_cost_latency(summary).axes[0],
        plot_round_chart(summary).axes[1],
        plot_progress(summary, VARIANT_ORDER[-1]).axes[0],
        plot_sim_vs_live(summary, live, recorded_live_note()).axes[0],
    ]
    both = pd.concat([costs, live["model_cost_per_page_usd"]])
    for axis in figs:
        low, high = axis.get_ylim()
        assert axis.get_yscale() == "log"
        assert high <= both.max() * 2.5 * 1.001
        assert low >= both.min() / 1.6 * 0.999


def test_line_value_labels_sit_on_a_surface_colored_box(
    summary: pd.DataFrame,
) -> None:
    for axis in (
        plot_round_chart(summary).axes[2],
        plot_progress(summary, VARIANT_ORDER[-1]).axes[2],
    ):
        assert axis.texts
        for text in axis.texts:
            patch = text.get_bbox_patch()
            assert patch is not None
            assert to_hex(patch.get_facecolor()) == PALETTE.surface


def test_change_titles_print_zero_not_negative_zero(summary: pd.DataFrame) -> None:
    flat = summary.copy()
    flat["model_cost_per_page_usd"] = 0.001
    flat["p50_latency_s"] = 1.0
    title = plot_progress(flat, VARIANT_ORDER[-1]).get_suptitle()
    assert "-0%" not in title
    assert "cost per page 0%" in title

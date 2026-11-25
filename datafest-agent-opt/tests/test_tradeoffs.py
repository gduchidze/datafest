"""Tests for the Decisions-versus-LLM comparison and calibration."""

import asyncio

import httpx2
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from doc_review_opt.agent.graph import (
    build_review_graph,
    review_checkpointer,
    run_batch,
)
from doc_review_opt.agent.routing import DEFAULT_POLICY, RoutingPolicy
from doc_review_opt.agent.variants import CALIBRATED_POLICY
from doc_review_opt.config import get_settings
from doc_review_opt.experiment.live import LIVE_PAGE_LIMIT, BudgetTracker
from doc_review_opt.experiment.runs import ReviewRun, scripted_attorney_decisions
from doc_review_opt.experiment.simulation import (
    SimulatedRedactor,
    SimulatedTypedClassifier,
)
from doc_review_opt.experiment.tradeoffs import (
    STEP_MATRIX,
    WHEN_TO_USE,
    NoFeasiblePolicyError,
    calibrate_policy,
    cascade_analysis,
    engine_comparison_table,
    evaluate_policy,
    live_engine_comparison,
    plot_calibration,
    plot_engine_tradeoff,
    plot_threshold_sweep,
    policy_report,
    replay,
    simulate_engine_comparison,
    threshold_sweep,
)
from doc_review_opt.fixtures import CALIBRATION, EVAL_SET, HOLDOUT
from doc_review_opt.models import ClassifierResult
from doc_review_opt.reporting.chart_style import PALETTE
from tests.helpers import assert_clean_chart, at, bar_colors, row
from tests.test_decisions_classifier import OK_ANSWERS, client_for, decisions_body

LIVE = get_settings().live
RUN_PAGE_RESERVE_USD = LIVE.per_page_ceiling_usd + LIVE.redaction_call_ceiling_usd


@pytest.fixture(scope="module")
def runs() -> dict[str, ReviewRun]:
    return asyncio.run(simulate_engine_comparison())


def classifications(runs: dict[str, ReviewRun], engine: str = "decisions") -> dict:  # type: ignore[type-arg]
    return {r.doc_id: r.classification for r in runs[engine].outcome.results}


def test_step_matrix_and_when_to_use_cover_the_brief() -> None:
    steps = {row["step"] for row in STEP_MATRIX}
    assert {
        "Responsiveness rubric",
        "Privilege predicate",
        "PII presence",
        "Near-duplicate check",
    } <= steps
    assert {
        "Escalation gate",
        "Redaction rewrite",
        "Privilege-log drafting",
        "QA tags",
    } <= steps
    assert {row["fits"] for row in STEP_MATRIX} == {"Decisions", "LLM", "Code"}
    assert all(
        {"factor", "use_decisions_when", "use_an_llm_when"} <= set(r)
        for r in WHEN_TO_USE
    )


def test_engine_table_columns_and_values(runs: dict[str, ReviewRun]) -> None:
    table = engine_comparison_table(runs).set_index("engine")
    assert list(table.index) == ["decisions", "llm-luna", "llm-terra", "agent"]
    assert {
        "accuracy_vs_labels",
        "agreement_vs_decisions",
        "cost_per_page_usd",
        "cost_per_100k_pages_usd",
        "p50_latency_s",
        "mean_sync_tokens",
        "escalation_rate",
        "privileged_leaks",
        "pii_leaks",
    } <= set(table.columns)
    assert at(table, "decisions", "agreement_vs_decisions") == 1.0
    assert at(table, "decisions", "cost_per_page_usd") < at(
        table, "llm-terra", "cost_per_page_usd"
    )
    assert at(table, "decisions", "p50_latency_s") < at(
        table, "llm-terra", "p50_latency_s"
    )
    assert at(table, "decisions", "escalation_rate") > at(
        table, "llm-terra", "escalation_rate"
    )
    fig = plot_engine_tradeoff(table.reset_index())
    assert_clean_chart(fig, 1)
    assert fig.axes[0].get_xscale() == "log"
    colors = {to_hex(np.asarray(c.get_facecolor())[0]) for c in fig.axes[0].collections}
    assert colors == {PALETTE.blue, PALETTE.orange, PALETTE.aqua, PALETTE.yellow}
    title = fig.get_suptitle()
    assert "Simulated" in title
    assert "both" not in title
    assert "live gpt-6-luna LLM was cheaper" in title
    legend = fig.axes[0].get_legend()
    assert legend is not None
    legend_names = {t.get_text() for t in legend.get_texts()}
    point_names = {t.get_text() for t in fig.axes[0].texts}
    assert legend_names <= point_names
    assert "gpt-6.1-terra agent" in legend_names


def test_threshold_sweep_shows_leaks_appear_at_high_thresholds(
    runs: dict[str, ReviewRun],
) -> None:
    sweep = threshold_sweep(runs, EVAL_SET)
    assert set(sweep["engine"]) == {"decisions", "llm-luna", "llm-terra"}
    luna = sweep[sweep["engine"] == "llm-luna"].set_index("privilege_threshold")
    assert at(luna, 0.2, "privileged_leaks") == 0
    assert at(luna, 0.9, "privileged_leaks") > 0
    assert at(luna, 0.9, "escalation_rate") < at(luna, 0.05, "escalation_rate")


def test_calibration_is_feasible_and_matches_the_shipped_policy(
    runs: dict[str, ReviewRun],
) -> None:
    results = classifications(runs)
    policy = calibrate_policy(results, CALIBRATION)
    assert policy == CALIBRATED_POLICY
    for pages in (CALIBRATION, HOLDOUT):
        score = evaluate_policy(results, pages, policy)
        assert (score.privileged_leaks, score.pii_leaks) == (0, 0)
    holdout_default = evaluate_policy(results, HOLDOUT, DEFAULT_POLICY)
    holdout_tuned = evaluate_policy(results, HOLDOUT, policy)
    assert holdout_tuned.escalation_rate < holdout_default.escalation_rate
    assert (
        holdout_tuned.blended_cost_per_page_usd
        < holdout_default.blended_cost_per_page_usd
    )


def test_calibration_raises_when_no_policy_is_leak_free(
    runs: dict[str, ReviewRun],
) -> None:
    grid = get_settings().calibration.model_copy(update={"pii_redaction": (0.99,)})
    with pytest.raises(NoFeasiblePolicyError, match="zero privileged and PII"):
        calibrate_policy(classifications(runs), CALIBRATION, grid)


def test_policy_report_covers_each_split_and_policy(runs: dict[str, ReviewRun]) -> None:
    report = policy_report(
        classifications(runs),
        {"calibration": CALIBRATION, "holdout": HOLDOUT},
        {"default": DEFAULT_POLICY, "calibrated": CALIBRATED_POLICY},
    )
    assert len(report) == 4
    assert {"escalation_rate", "privileged_leaks", "pii_leaks"} <= set(report.columns)


async def test_replay_matches_the_graph() -> None:
    classifier = SimulatedTypedClassifier("llm-luna")
    graph = build_review_graph(classifier, SimulatedRedactor(), review_checkpointer())
    outcome = await run_batch(
        graph,
        EVAL_SET,
        thread_id="replay",
        attorney_decisions=scripted_attorney_decisions(EVAL_SET),
    )
    for result in outcome.results:
        page = next(p for p in EVAL_SET if p.doc_id == result.doc_id)
        assert result.classification is not None
        replayed = replay(page, result.classification, DEFAULT_POLICY).outcome
        assert (replayed.route, replayed.decision, replayed.escalated) == (
            result.route,
            result.decision,
            result.escalated,
        )
        assert replayed.produced_text == result.produced_text


async def test_cascade_beats_decisions_only_on_quality_and_terra_only_on_cost() -> None:
    table = (await cascade_analysis()).set_index("arm")
    cascade, only, terra = (
        row(table, "Decisions + terra cascade"),
        row(table, "Decisions only"),
        row(table, "Terra only"),
    )
    assert cascade["mean_quality"] > only["mean_quality"]
    assert cascade["attorney_escalations"] < only["attorney_escalations"]
    assert cascade["model_cost_per_page_usd"] < terra["model_cost_per_page_usd"]
    assert cascade["blended_cost_per_page_usd"] < only["blended_cost_per_page_usd"]
    assert isinstance(table, pd.DataFrame)


def test_policy_is_a_frozen_dataclass() -> None:
    assert RoutingPolicy().privilege_escalation == 0.2


async def test_live_engine_comparison_returns_partial_runs_when_the_budget_ends() -> (
    None
):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=decisions_body(OK_ANSWERS))

    pages = [p for p in EVAL_SET if p.doc_id not in {"ARC-000572", "ARC-000580"}]
    budget = BudgetTracker(limit_usd=1.0)
    budget.add(1.0 - LIVE_PAGE_LIMIT * RUN_PAGE_RESERVE_USD - 0.0001)
    runs = await live_engine_comparison(
        pages, client=client_for(handler), budget=budget
    )
    assert list(runs) == ["decisions"]


def test_replay_redacts_an_escalated_page_whose_classifier_errored() -> None:
    secret = next(
        p
        for p in EVAL_SET
        if p.labels.pii_spans
        and scripted_attorney_decisions([p])[p.doc_id] == "produce"
    )
    failed = ClassifierResult(engine="x", model="m", error="boom")
    replayed = replay(secret, failed, DEFAULT_POLICY)
    assert replayed.outcome.produced_text is not None
    assert not [
        s for s in secret.labels.pii_spans if s in replayed.outcome.produced_text
    ]


def test_threshold_sweep_chart_marks_leaks_and_labels_lines(
    runs: dict[str, ReviewRun],
) -> None:
    sweep = threshold_sweep(runs, EVAL_SET)
    fig = plot_threshold_sweep(sweep)
    assert_clean_chart(fig, 1)
    lines = {to_hex(line.get_color()) for line in fig.axes[0].lines}
    assert {PALETTE.blue, PALETTE.orange, PALETTE.aqua} <= lines
    assert PALETTE.critical in lines
    texts = {t.get_text() for t in fig.axes[0].texts}
    assert {"Decisions", "gpt-6-luna LLM", "gpt-6.1-terra LLM"} <= texts
    assert fig.axes[0].get_xlim() == (0.0, 1.0)
    title = fig.get_suptitle()
    assert title.startswith("Simulated:")
    assert "gpt-6-luna LLM leaks privileged pages from 0.50" in title
    assert "gpt-6.1-terra LLM from 0.80" in title
    assert "Decisions never leaks" in title


def test_threshold_sweep_chart_without_leaks_says_so() -> None:
    sweep = pd.DataFrame(
        {
            "engine": ["decisions", "decisions"],
            "privilege_threshold": [0.1, 0.2],
            "escalation_rate": [0.5, 0.4],
            "privileged_leaks": [0, 0],
        }
    )
    title = plot_threshold_sweep(sweep).get_suptitle()
    assert "no privileged page leaks" in title
    assert "Simulated" in title


def test_calibration_chart_reports_zero_leaks(runs: dict[str, ReviewRun]) -> None:
    table = policy_report(
        classifications(runs),
        {"calibration": CALIBRATION, "holdout": HOLDOUT},
        {"default": DEFAULT_POLICY, "calibrated": CALIBRATED_POLICY},
    )
    fig = plot_calibration(table)
    assert_clean_chart(fig, 1)
    assert bar_colors(fig.axes[0])[0] == PALETTE.yellow
    assert bar_colors(fig.axes[0])[-1] == PALETTE.aqua
    assert "0 leaks" in fig.get_suptitle()

"""Tests for the deterministic per-variant metrics and the calibration holdout check."""

from __future__ import annotations

import asyncio
import copy
import tomllib
from importlib.resources import files
from typing import Any, Literal

import pytest

from doc_review_opt.agent.variants import (
    BASELINE,
    CALIBRATED_POLICY,
    VARIANT_ORDER,
)
from doc_review_opt.config import get_settings, parse_settings
from doc_review_opt.exceptions import ConfigError
from doc_review_opt.experiment.metrics import (
    ROUTES,
    all_variant_metrics,
    classifications_by_page,
    holdout_check,
    variant_metrics,
)
from doc_review_opt.experiment.simulation import simulate_all
from doc_review_opt.experiment.traces import Trace
from doc_review_opt.experiment.tradeoffs import (
    evaluate_policy,
    simulate_engine_comparison,
)
from doc_review_opt.fixtures import EVAL_SET, split_pages
from doc_review_opt.models import ClassifierResult, PageLabels, ReviewPage, Route
from tests.helpers import make_result

FLOORS = get_settings().metrics
VARIANT = "v"
SPAN_KEPT = "111-22-3333"
SPAN_REMOVED = "jane@example.com"


def page(
    doc_id: str,
    route: Route,
    *,
    responsive: Literal[0, 1, 2] = 2,
    privileged: bool = False,
    spans: tuple[str, ...] = (),
) -> ReviewPage:
    disposition: Literal["produce", "withhold", "attorney_review"] = (
        "attorney_review"
        if route == "attorney_review"
        else "withhold"
        if route == "withhold"
        else "produce"
    )
    return ReviewPage(
        doc_id=doc_id,
        custodian="c",
        author="a",
        text=f"page {doc_id}",
        labels=PageLabels(
            expected_disposition=disposition,
            expected_route=route,
            responsive_level=responsive,
            privileged=privileged,
            contains_pii=bool(spans),
            pii_spans=spans,
            must_escalate=route == "attorney_review",
            difficulty="routine",
            risk="low",
        ),
    )


def trace(
    pages: dict[str, ReviewPage],
    doc_id: str,
    route: Route,
    *,
    latency: float,
    cost: float,
    produced_text: str | None = None,
    escalated: bool = False,
) -> Trace:
    labels = pages[doc_id].labels
    produced = produced_text is not None
    return Trace(
        variant=VARIANT,
        variant_label="Variant",
        mode="simulated",
        doc_id=doc_id,
        difficulty="routine",
        risk="low",
        engine="stub",
        models_used=("stub",),
        routing_tokens=0,
        tool_calls=0,
        expected_tools=(),
        tools=(),
        route=route,
        route_reason="",
        expected_route=labels.expected_route,
        decision="produce" if produced else "withhold",
        expected_disposition=labels.expected_disposition,
        escalated=escalated,
        must_escalate=labels.must_escalate,
        attorney_decision=None,
        input_tokens=0,
        cached_tokens=0,
        cache_write_tokens=0,
        output_tokens=0,
        visible_output_tokens=0,
        reasoning_tokens=0,
        sync_tokens=0,
        background_tokens=0,
        total_tokens=0,
        latency_s=latency,
        sync_cost_usd=cost,
        background_cost_usd=0.0,
        model_cost_usd=cost,
        attorney_cost_usd=0.0,
        cost_usd=cost,
        produced_text=produced_text,
        privileged_leak=False,
        pii_leak=False,
        route_correct=route == labels.expected_route,
        escalation_correct=escalated == labels.must_escalate,
        unnecessary_escalation=escalated and not labels.must_escalate,
        responsive_withheld=False,
        extra_tool_calls=0,
        parse_failed=False,
        quality_score=0.9,
        policy_compliant=True,
    )


PAGES = {
    p.doc_id: p
    for p in (
        page("P1", "produce"),
        page("P2", "redact", spans=(SPAN_KEPT, SPAN_REMOVED)),
        page("P3", "produce", responsive=1),
        page("P4", "attorney_review", privileged=True),
        page("P5", "attorney_review", privileged=True),
        page("P6", "withhold", responsive=0),
    )
}
TRACES = [
    trace(PAGES, "P1", "produce", latency=1.0, cost=0.1, produced_text="clean"),
    trace(
        PAGES,
        "P2",
        "redact",
        latency=2.0,
        cost=0.2,
        produced_text=f"call [X] at {SPAN_KEPT}",
    ),
    trace(PAGES, "P3", "withhold", latency=3.0, cost=0.3),
    trace(PAGES, "P4", "attorney_review", latency=4.0, cost=0.4, escalated=True),
    trace(PAGES, "P5", "withhold", latency=5.0, cost=0.5),
    trace(PAGES, "P6", "attorney_review", latency=6.0, cost=0.6, escalated=True),
]


def subset(*doc_ids: str) -> list[Trace]:
    return [t for t in TRACES if t.doc_id in doc_ids]


def test_route_accuracy_and_confusion_matrix() -> None:
    metrics = variant_metrics(TRACES, list(PAGES.values()), VARIANT)
    assert ROUTES == ("produce", "withhold", "redact", "attorney_review")
    assert metrics.pages == 6
    assert metrics.route_accuracy == pytest.approx(3 / 6)
    assert metrics.confusion == (
        (1, 1, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 1, 0, 1),
    )


def test_per_route_precision_recall_f1() -> None:
    metrics = variant_metrics(TRACES, list(PAGES.values()), VARIANT)
    by_route = {score.route: score for score in metrics.route_scores}
    produce = by_route["produce"]
    assert (produce.support, produce.predicted) == (2, 1)
    assert (produce.precision, produce.recall) == (1.0, 0.5)
    assert produce.f1 == pytest.approx(2 / 3)
    withhold = by_route["withhold"]
    assert (withhold.precision, withhold.recall, withhold.f1) == (0.0, 0.0, 0.0)
    redact = by_route["redact"]
    assert (redact.precision, redact.recall, redact.f1) == (1.0, 1.0, 1.0)
    attorney = by_route["attorney_review"]
    assert (attorney.precision, attorney.recall, attorney.f1) == (0.5, 0.5, 0.5)


def test_relevance_recall_and_over_withholding() -> None:
    metrics = variant_metrics(TRACES, list(PAGES.values()), VARIANT)
    assert metrics.responsive_pages == 3
    assert metrics.over_withheld_pages == 1
    assert metrics.over_withholding_rate == pytest.approx(1 / 3)
    assert metrics.relevance_recall == pytest.approx(2 / 3)


def test_responsive_duplicate_labelled_withhold_is_not_over_withholding() -> None:
    duplicate = page("P7", "withhold", responsive=2)
    pages = [*PAGES.values(), duplicate]
    withheld = trace(
        {**PAGES, "P7": duplicate}, "P7", "withhold", latency=1.0, cost=0.1
    )
    metrics = variant_metrics([*TRACES, withheld], pages, VARIANT)
    assert metrics.responsive_pages == 3
    assert metrics.over_withheld_pages == 1


def test_privileged_recall_counts_only_attorney_review_routes() -> None:
    metrics = variant_metrics(TRACES, list(PAGES.values()), VARIANT)
    assert metrics.privileged_pages == 2
    assert metrics.privileged_recall == 0.5


def test_pii_recall_counts_spans_absent_from_produced_text() -> None:
    metrics = variant_metrics(TRACES, list(PAGES.values()), VARIANT)
    assert metrics.pii_spans == 2
    assert metrics.pii_recall == 0.5
    assert metrics.pii_leaked_pages == 1


def test_escalation_error_rates() -> None:
    metrics = variant_metrics(TRACES, list(PAGES.values()), VARIANT)
    assert metrics.under_escalation_rate == 0.5
    assert metrics.unnecessary_escalation_rate == 0.25


def test_latency_percentiles_and_cost_per_page() -> None:
    metrics = variant_metrics(TRACES, list(PAGES.values()), VARIANT)
    assert metrics.p50_latency_s == pytest.approx(3.5)
    assert metrics.p95_latency_s == pytest.approx(5.75)
    assert metrics.cost_per_page_usd == pytest.approx(0.35)


def test_metrics_below_their_floor_are_watched() -> None:
    metrics = variant_metrics(TRACES, list(PAGES.values()), VARIANT)
    assert set(metrics.watch) == {
        "route_accuracy",
        "relevance_recall",
        "privileged_recall",
        "pii_recall",
        "under_escalation_rate",
        "unnecessary_escalation_rate",
    }


def test_a_clean_subset_has_no_watch_flags() -> None:
    metrics = variant_metrics(subset("P1"), list(PAGES.values()), VARIANT)
    assert metrics.watch == ()
    assert metrics.route_accuracy == 1.0


def test_subset_without_privileged_or_pii_pages_is_defined() -> None:
    metrics = variant_metrics(subset("P1", "P3"), list(PAGES.values()), VARIANT)
    assert metrics.privileged_pages == 0
    assert metrics.privileged_recall == 1.0
    assert metrics.pii_spans == 0
    assert metrics.pii_recall == 1.0
    assert metrics.under_escalation_rate == 0.0
    attorney = {s.route: s for s in metrics.route_scores}["attorney_review"]
    assert (attorney.support, attorney.predicted) == (0, 0)
    assert (attorney.precision, attorney.recall, attorney.f1) == (1.0, 1.0, 1.0)


def test_subset_without_responsive_pages_is_defined() -> None:
    metrics = variant_metrics(subset("P4", "P5", "P6"), list(PAGES.values()), VARIANT)
    assert metrics.responsive_pages == 0
    assert metrics.relevance_recall == 1.0
    assert metrics.over_withholding_rate == 0.0


def test_subset_with_only_attorney_pages_has_no_false_alarm_base() -> None:
    metrics = variant_metrics(subset("P4", "P5"), list(PAGES.values()), VARIANT)
    assert metrics.unnecessary_escalation_rate == 0.0


def test_no_traces_for_a_variant_is_an_error() -> None:
    with pytest.raises(ValueError, match="no traces"):
        variant_metrics(TRACES, list(PAGES.values()), "missing")


def test_variant_metrics_ignore_other_variants() -> None:
    other = TRACES[0].model_copy(update={"variant": "other", "route": "withhold"})
    metrics = variant_metrics([*TRACES, other], list(PAGES.values()), VARIANT)
    assert metrics.pages == 6


@pytest.fixture(scope="module")
def traces() -> list[Trace]:
    return asyncio.run(simulate_all())


def test_all_variant_metrics_follow_the_requested_order(traces: list[Trace]) -> None:
    result = all_variant_metrics(traces, VARIANT_ORDER)
    assert [m.variant for m in result] == VARIANT_ORDER
    assert all(m.pages == len(EVAL_SET) for m in result)


def test_new_metrics_agree_with_the_existing_trace_columns(
    traces: list[Trace],
) -> None:
    for metrics in all_variant_metrics(traces, VARIANT_ORDER):
        own = [t for t in traces if t.variant == metrics.variant]
        assert metrics.route_accuracy == pytest.approx(
            sum(t.route_correct for t in own) / len(own)
        )
        assert metrics.pii_leaked_pages == sum(t.pii_leak for t in own)
        assert metrics.cost_per_page_usd == pytest.approx(
            sum(t.cost_usd for t in own) / len(own)
        )


def test_baseline_leaks_pii_and_the_metric_shows_it(traces: list[Trace]) -> None:
    metrics = all_variant_metrics(traces, [BASELINE])[0]
    assert metrics.pii_leaked_pages == 2
    assert metrics.pii_recall < 1.0
    assert "pii_recall" in metrics.watch


def test_split_is_deterministic_disjoint_covering_and_stratified() -> None:
    first = split_pages(EVAL_SET)
    assert first == split_pages(EVAL_SET)
    calibration, holdout = first
    ids = [p.doc_id for p in (*calibration, *holdout)]
    assert len(ids) == len(set(ids)) == len(EVAL_SET)
    assert set(ids) == {p.doc_id for p in EVAL_SET}
    for route in ROUTES:
        in_calibration = sum(p.labels.expected_route == route for p in calibration)
        in_holdout = sum(p.labels.expected_route == route for p in holdout)
        assert abs(in_calibration - in_holdout) <= 1


def small_world() -> tuple[tuple[ReviewPage, ...], dict[str, ClassifierResult]]:
    pages = (
        page("H1", "produce"),
        page("H2", "produce"),
        page("H3", "attorney_review", privileged=True),
        page("H4", "attorney_review", privileged=True),
    )
    clean = make_result(responsive=2.0, confidence=0.9, privileged=0.01, pii=0.01)
    risky = make_result(responsive=2.0, confidence=0.9, privileged=0.95, pii=0.01)
    return pages, {"H1": clean, "H2": clean, "H3": risky, "H4": risky}


def test_holdout_check_scores_both_halves_on_hand_made_pages() -> None:
    pages, results = small_world()
    report = holdout_check(results, pages)
    assert (report.calibration_pages, report.holdout_pages) == (2, 2)
    assert report.holds
    refit = {s.split: s.score for s in report.scores if s.policy == "refit"}
    assert set(refit) == {"calibration", "holdout"}
    for score in refit.values():
        assert score.escalation_rate == 0.5
        assert (score.privileged_leaks, score.pii_leaks) == (0, 0)


def test_holdout_check_flags_a_leak_on_unseen_pages() -> None:
    pages, results = small_world()
    leaky = dict(results)
    leaky["H4"] = make_result(responsive=2.0, confidence=0.9, privileged=0.01, pii=0.01)
    report = holdout_check(leaky, pages)
    assert not report.holds


def test_holdout_check_on_the_eval_set_matches_direct_scoring() -> None:
    runs = asyncio.run(simulate_engine_comparison())
    clean = classifications_by_page(runs["decisions"])
    report = holdout_check(clean, EVAL_SET)
    calibration, holdout = split_pages(EVAL_SET)
    assert report.refit_policy == CALIBRATED_POLICY
    assert report.holds
    by_key = {(s.policy, s.split): s.score for s in report.scores}
    assert by_key[("refit", "holdout")] == evaluate_policy(
        clean, holdout, CALIBRATED_POLICY
    )
    assert by_key[("refit", "calibration")] == evaluate_policy(
        clean, calibration, CALIBRATED_POLICY
    )
    assert {policy for policy, _ in by_key} == {"default", "refit"}


def raw() -> dict[str, Any]:
    text = files("doc_review_opt").joinpath("config.toml").read_text("utf-8")
    return tomllib.loads(text)


def test_metric_floors_load_from_the_toml() -> None:
    assert 0 < FLOORS.median_quantile < FLOORS.tail_quantile <= 1
    assert FLOORS.privileged_recall_floor == 1.0


def test_metric_floor_outside_zero_to_one_is_rejected() -> None:
    broken = copy.deepcopy(raw())
    broken["metrics"]["route_accuracy_floor"] = 1.5
    with pytest.raises(ConfigError, match=r"metrics\.route_accuracy_floor"):
        parse_settings(broken)


def test_classifications_by_page_covers_every_page_of_a_run() -> None:
    runs = asyncio.run(simulate_engine_comparison())
    found = classifications_by_page(runs["decisions"])
    assert set(found) == {p.doc_id for p in EVAL_SET}
    assert {c.engine for c in found.values()} == {"decisions"}

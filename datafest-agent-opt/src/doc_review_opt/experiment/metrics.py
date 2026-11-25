"""Deterministic reported metrics per variant and the calibration holdout check."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from doc_review_opt.agent.routing import DEFAULT_POLICY, RoutingPolicy
from doc_review_opt.config import get_settings
from doc_review_opt.experiment.runs import ReviewRun
from doc_review_opt.experiment.traces import Trace
from doc_review_opt.experiment.tradeoffs import (
    PolicyScore,
    calibrate_policy,
    evaluate_policy,
)
from doc_review_opt.fixtures import EVAL_SET, split_pages
from doc_review_opt.models import ClassifierResult, PageLabels, ReviewPage, Route

METRICS = get_settings().metrics
ROUTES: tuple[Route, ...] = ("produce", "withhold", "redact", "attorney_review")
FLOORS = {
    "route_accuracy": METRICS.route_accuracy_floor,
    "relevance_recall": METRICS.relevance_recall_floor,
    "privileged_recall": METRICS.privileged_recall_floor,
    "pii_recall": METRICS.pii_recall_floor,
}
CEILINGS = {
    "under_escalation_rate": METRICS.max_under_escalation_rate,
    "unnecessary_escalation_rate": METRICS.max_unnecessary_escalation_rate,
}
_WITHHOLD: Route = "withhold"
_ATTORNEY: Route = "attorney_review"


@dataclass(frozen=True)
class RouteScore:
    route: Route
    support: int
    predicted: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class VariantMetrics:
    variant: str
    pages: int
    route_accuracy: float
    route_scores: tuple[RouteScore, ...]
    confusion: tuple[tuple[int, ...], ...]
    responsive_pages: int
    over_withheld_pages: int
    relevance_recall: float
    over_withholding_rate: float
    privileged_pages: int
    privileged_recall: float
    pii_spans: int
    pii_recall: float
    pii_leaked_pages: int
    under_escalation_rate: float
    unnecessary_escalation_rate: float
    p50_latency_s: float
    p95_latency_s: float
    cost_per_page_usd: float
    watch: tuple[str, ...]


@dataclass(frozen=True)
class HoldoutScore:
    policy: str
    split: str
    score: PolicyScore


@dataclass(frozen=True)
class HoldoutReport:
    calibration_pages: int
    holdout_pages: int
    refit_policy: RoutingPolicy
    scores: tuple[HoldoutScore, ...]
    holds: bool


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _route_score(route: Route, confusion: Sequence[Sequence[int]]) -> RouteScore:
    index = ROUTES.index(route)
    hits = confusion[index][index]
    support = sum(confusion[index])
    predicted = sum(row[index] for row in confusion)
    precision = _ratio(hits, predicted)
    recall = _ratio(hits, support)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RouteScore(route, support, predicted, precision, recall, f1)


def _watch(values: Mapping[str, float]) -> tuple[str, ...]:
    below = [name for name, floor in FLOORS.items() if values[name] < floor]
    above = [name for name, ceiling in CEILINGS.items() if values[name] > ceiling]
    return tuple(below + above)


def _pii_counts(
    chosen: Sequence[Trace], labels: Mapping[str, PageLabels]
) -> tuple[int, int, int]:
    spans = 0
    leaked_spans = 0
    leaked_pages = 0
    for trace in chosen:
        text = trace.produced_text or ""
        found = sum(span in text for span in labels[trace.doc_id].pii_spans)
        spans += len(labels[trace.doc_id].pii_spans)
        leaked_spans += found
        leaked_pages += found > 0
    return spans, leaked_spans, leaked_pages


def _escalation_counts(chosen: Sequence[Trace]) -> tuple[int, int, int, int]:
    needs_attorney = [t for t in chosen if t.expected_route == _ATTORNEY]
    no_attorney = [t for t in chosen if t.expected_route != _ATTORNEY]
    missed = sum(not trace.escalated for trace in needs_attorney)
    unnecessary = sum(trace.escalated for trace in no_attorney)
    return missed, len(needs_attorney), unnecessary, len(no_attorney)


def variant_metrics(
    traces: Sequence[Trace], pages: Sequence[ReviewPage], variant: str
) -> VariantMetrics:
    chosen = [trace for trace in traces if trace.variant == variant]
    if not chosen:
        raise ValueError(f"no traces for variant {variant!r}")
    labels = {page.doc_id: page.labels for page in pages}
    count = len(chosen)

    confusion = [[0] * len(ROUTES) for _ in ROUTES]
    for trace in chosen:
        confusion[ROUTES.index(trace.expected_route)][ROUTES.index(trace.route)] += 1
    route_accuracy = sum(confusion[i][i] for i in range(len(ROUTES))) / count

    responsive = [
        trace
        for trace in chosen
        if labels[trace.doc_id].responsive_level >= 1
        and not labels[trace.doc_id].privileged
        and labels[trace.doc_id].expected_disposition != "withhold"
    ]
    over_withheld = sum(trace.route == _WITHHOLD for trace in responsive)

    privileged = [trace for trace in chosen if labels[trace.doc_id].privileged]
    privileged_caught = sum(trace.route == _ATTORNEY for trace in privileged)

    spans, leaked_spans, leaked_pages = _pii_counts(chosen, labels)
    missed, needs_attorney, unnecessary, no_attorney = _escalation_counts(chosen)

    latencies = [trace.latency_s for trace in chosen]
    values = {
        "route_accuracy": route_accuracy,
        "relevance_recall": _ratio(len(responsive) - over_withheld, len(responsive)),
        "privileged_recall": _ratio(privileged_caught, len(privileged)),
        "pii_recall": _ratio(spans - leaked_spans, spans),
        "under_escalation_rate": _rate(missed, needs_attorney),
        "unnecessary_escalation_rate": _rate(unnecessary, no_attorney),
    }
    return VariantMetrics(
        variant=variant,
        pages=count,
        route_accuracy=route_accuracy,
        route_scores=tuple(_route_score(route, confusion) for route in ROUTES),
        confusion=tuple(tuple(row) for row in confusion),
        responsive_pages=len(responsive),
        over_withheld_pages=over_withheld,
        relevance_recall=values["relevance_recall"],
        over_withholding_rate=_rate(over_withheld, len(responsive)),
        privileged_pages=len(privileged),
        privileged_recall=values["privileged_recall"],
        pii_spans=spans,
        pii_recall=values["pii_recall"],
        pii_leaked_pages=leaked_pages,
        under_escalation_rate=values["under_escalation_rate"],
        unnecessary_escalation_rate=values["unnecessary_escalation_rate"],
        p50_latency_s=float(np.quantile(latencies, METRICS.median_quantile)),
        p95_latency_s=float(np.quantile(latencies, METRICS.tail_quantile)),
        cost_per_page_usd=sum(trace.cost_usd for trace in chosen) / count,
        watch=_watch(values),
    )


def all_variant_metrics(
    traces: Sequence[Trace],
    variants: Sequence[str],
    pages: Sequence[ReviewPage] = EVAL_SET,
) -> list[VariantMetrics]:
    return [variant_metrics(traces, pages, variant) for variant in variants]


def classifications_by_page(run: ReviewRun) -> dict[str, ClassifierResult]:
    found: dict[str, ClassifierResult] = {}
    for result in run.outcome.results:
        if result.classification is None:
            raise ValueError(f"{result.doc_id}: result has no classification")
        found[result.doc_id] = result.classification
    return found


def holdout_check(
    results: Mapping[str, ClassifierResult],
    pages: Sequence[ReviewPage] = EVAL_SET,
) -> HoldoutReport:
    calibration, holdout = split_pages(tuple(pages))
    refit = calibrate_policy(results, calibration)
    policies = {"default": DEFAULT_POLICY, "refit": refit}
    halves = {"calibration": calibration, "holdout": holdout}
    scores = tuple(
        HoldoutScore(name, split, evaluate_policy(results, half, policy))
        for name, policy in policies.items()
        for split, half in halves.items()
    )
    unseen = [s.score for s in scores if s.split == "holdout" and s.policy == "refit"]
    return HoldoutReport(
        calibration_pages=len(calibration),
        holdout_pages=len(holdout),
        refit_policy=refit,
        scores=scores,
        holds=all(s.privileged_leaks == 0 and s.pii_leaks == 0 for s in unseen),
    )

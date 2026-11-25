"""Small typed helpers that keep each notebook round to one import."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from doc_review_opt.agent.variants import VARIANT_LABELS, VARIANT_ORDER
from doc_review_opt.config import get_settings
from doc_review_opt.experiment.metrics import (
    CEILINGS,
    FLOORS,
    ROUTES,
    HoldoutReport,
    VariantMetrics,
    all_variant_metrics,
    variant_metrics,
)
from doc_review_opt.experiment.simulation import variant_request_example
from doc_review_opt.experiment.traces import Trace, deterministic_guardrail_check
from doc_review_opt.fixtures import ATTORNEYS, EVAL_SET
from doc_review_opt.models import PageResult, ReviewPage
from doc_review_opt.reporting.chart_style import ROUTE_LABELS

_TITLE_SEPARATOR = " -- "
_ATTORNEY_TAG = " (attorney)"


def _title(page: ReviewPage) -> str:
    return page.text.partition("\n")[0].split(_TITLE_SEPARATOR, 1)[-1]


def _sender(page: ReviewPage) -> str:
    return page.author + (_ATTORNEY_TAG if page.author in ATTORNEYS else "")


def page_table(
    pages: Sequence[ReviewPage], results: Sequence[PageResult] = ()
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "doc_id": [page.doc_id for page in pages],
            "page": [_title(page) for page in pages],
            "from": [_sender(page) for page in pages],
            "expected": [page.labels.expected_route for page in pages],
        }
    )
    if results:
        by_id = {result.doc_id: result for result in results}
        table["route"] = [by_id[page.doc_id].route for page in pages]
        table["decision"] = [by_id[page.doc_id].decision for page in pages]
    return table


@dataclass(frozen=True)
class GateResult:
    variant: str
    label: str
    pass_rate: float
    accepted: bool
    failures: dict[str, int]
    privileged_leaks: int
    pii_leaks: int

    def __str__(self) -> str:
        verdict = "ACCEPTED" if self.accepted else "REJECTED"
        floor = get_settings().quality.guardrail_pass_floor
        failures = ", ".join(f"{k}={v}" for k, v in self.failures.items()) or "none"
        return (
            f"{self.label}: {verdict} (guardrail pass {self.pass_rate:.0%}, "
            f"floor {floor:.0%}; failures: {failures})"
        )


def _variant_traces(traces: Sequence[Trace], variant: str) -> list[Trace]:
    return [trace for trace in traces if trace.variant == variant]


def gate(
    traces: Sequence[Trace], variant: str, pages: Sequence[ReviewPage] = EVAL_SET
) -> GateResult:
    by_id = {page.doc_id: page for page in pages}
    chosen = _variant_traces(traces, variant)
    codes = [deterministic_guardrail_check(by_id[t.doc_id], t) for t in chosen]
    failures = Counter(code for page_codes in codes for code in page_codes)
    failed_pages = sum(1 for page_codes in codes if page_codes)
    rate = 1 - failed_pages / len(chosen)
    privileged = sum(t.privileged_leak for t in chosen)
    pii = sum(t.pii_leak for t in chosen)
    floor = get_settings().quality.guardrail_pass_floor
    return GateResult(
        variant=variant,
        label=VARIANT_LABELS[variant],
        pass_rate=rate,
        accepted=privileged == 0 and pii == 0 and rate >= floor,
        failures=dict(failures),
        privileged_leaks=privileged,
        pii_leaks=pii,
    )


def gate_table(
    traces: Sequence[Trace],
    variants: Sequence[str],
    pages: Sequence[ReviewPage] = EVAL_SET,
) -> pd.DataFrame:
    results = [gate(traces, variant, pages) for variant in variants]
    return pd.DataFrame(
        {
            "label": [r.label for r in results],
            "pass_rate": [r.pass_rate for r in results],
            "accepted": [r.accepted for r in results],
        }
    )


def _value(summary: pd.DataFrame, variant: str, column: str) -> float:
    return float(summary.loc[summary["variant"] == variant, column].iloc[0])


def symptoms(summary: pd.DataFrame, variant: str) -> str:
    def value(column: str) -> float:
        return _value(summary, variant, column)

    return "\n".join(
        (
            f"{VARIANT_LABELS[variant]}",
            f"  tool calls: {value('mean_tool_calls'):.1f} per page "
            f"({value('mean_extra_tool_calls'):.1f} unneeded)",
            f"  sync tokens: {value('mean_sync_tokens'):,.0f} per page",
            f"  p50 latency: {value('p50_latency_s'):.2f}s",
            f"  model cost: ${value('model_cost_per_page_usd'):.5f} per page",
            f"  escalation rate: {value('escalation_rate'):.0%}",
            f"  leaks: privileged {int(value('privileged_leaks'))}, "
            f"PII {int(value('pii_leaks'))}",
        )
    )


def _request_facts(request: Mapping[str, Any]) -> dict[str, object]:
    text = request.get("text", {})
    choice = request.get("tool_choice", "-")
    return {
        "model": request["model"],
        "reasoning": request.get("reasoning", {}).get("effort", "-"),
        "verbosity": text.get("verbosity", "-"),
        "structured_output": text.get("format", {}).get("name", "free text"),
        "max_output_tokens": request.get("max_output_tokens", "-"),
        "tool_choice": choice if isinstance(choice, str) else json.dumps(choice),
        "tools": ", ".join(sorted(t["name"] for t in request.get("tools", []))) or "-",
        "questions": ", ".join(q["name"] for q in request.get("questions", [])) or "-",
        "prompt_cache_key": request.get("prompt_cache_key", "-"),
        "instructions_chars": len(request.get("instructions", "")),
        "input_chars": len(json.dumps(request["input"])),
    }


def request_diff(before: str, after: str, page: ReviewPage) -> pd.DataFrame:
    old = _request_facts(variant_request_example(before, page))
    new = _request_facts(variant_request_example(after, page))
    rows = [
        {"setting": key, "before": str(old[key]), "after": str(new[key])}
        for key in old
        if old[key] != new[key]
    ]
    return pd.DataFrame(rows, columns=["setting", "before", "after"])


_METRICS = (
    ("quality", "mean_quality", "{:.2f}"),
    ("privileged leaks", "privileged_leaks", "{:d}"),
    ("PII leaks", "pii_leaks", "{:d}"),
    ("escalation rate", "escalation_rate", "{:.0%}"),
    ("tool calls per page", "mean_tool_calls", "{:.1f}"),
    ("sync tokens", "mean_sync_tokens", "{:,.0f}"),
    ("p50 latency (s)", "p50_latency_s", "{:.2f}"),
    ("model $/page", "model_cost_per_page_usd", "${:.5f}"),
    ("cost with attorney time $/page", "cost_per_page_usd", "${:.3f}"),
)
_INT_COLUMNS = ("privileged_leaks", "pii_leaks")


def before_after(summary: pd.DataFrame, previous: str, current: str) -> pd.DataFrame:
    def cell(variant: str, column: str, pattern: str) -> str:
        value = _value(summary, variant, column)
        return pattern.format(int(value) if column in _INT_COLUMNS else value)

    return pd.DataFrame(
        {
            "metric": label,
            "before": cell(previous, column, pattern),
            "after": cell(current, column, pattern),
        }
        for label, column, pattern in _METRICS
    )


_WATCH_MARK = " (watch)"
_REPORTED = (
    ("route accuracy", "route_accuracy", "{:.0%}", "route_accuracy"),
    ("relevance recall", "relevance_recall", "{:.0%}", "relevance_recall"),
    (
        "over-withholding rate",
        "over_withholding_rate",
        "{:.0%}",
        "relevance_recall",
    ),
    ("privileged recall", "privileged_recall", "{:.0%}", "privileged_recall"),
    ("PII recall", "pii_recall", "{:.0%}", "pii_recall"),
    (
        "under-escalation rate",
        "under_escalation_rate",
        "{:.0%}",
        "under_escalation_rate",
    ),
    (
        "unnecessary escalation rate",
        "unnecessary_escalation_rate",
        "{:.0%}",
        "unnecessary_escalation_rate",
    ),
    ("p50 latency (s)", "p50_latency_s", "{:.2f}", ""),
    ("p95 latency (s)", "p95_latency_s", "{:.2f}", ""),
    ("$/page with attorney time", "cost_per_page_usd", "${:.3f}", ""),
)


def watch_percent(value: float, bound: float, *, below: bool) -> str:
    shown = round(value * 100)
    limit = round(bound * 100)
    if below and shown >= limit:
        shown = math.floor(value * 100)
    elif not below and shown <= limit:
        shown = math.ceil(value * 100)
    return f"{shown}%"


def _bound(attribute: str) -> tuple[float, bool] | None:
    if attribute in FLOORS:
        return FLOORS[attribute], True
    if attribute in CEILINGS:
        return CEILINGS[attribute], False
    if attribute == "over_withholding_rate":
        return 1 - FLOORS["relevance_recall"], False
    return None


def _reported_cell(
    metrics: VariantMetrics, attribute: str, pattern: str, watch_key: str
) -> str:
    value = getattr(metrics, attribute)
    flagged = watch_key in metrics.watch
    bound = _bound(attribute)
    if flagged and bound is not None:
        return watch_percent(value, bound[0], below=bound[1]) + _WATCH_MARK
    return pattern.format(value) + (_WATCH_MARK if flagged else "")


def metrics_table(
    traces: Sequence[Trace],
    variants: Sequence[str],
    pages: Sequence[ReviewPage] = EVAL_SET,
) -> pd.DataFrame:
    floor = get_settings().quality.guardrail_pass_floor
    gates = [gate(traces, variant, pages) for variant in variants]
    metrics = all_variant_metrics(traces, variants, pages)
    labels = [VARIANT_LABELS[variant] for variant in variants]
    rows: list[tuple[str, str, list[str]]] = [
        (
            "hard gate",
            "privileged pages produced (must be 0)",
            [str(g.privileged_leaks) for g in gates],
        ),
        (
            "hard gate",
            "PII leaked pages (must be 0)",
            [str(g.pii_leaks) for g in gates],
        ),
        (
            "hard gate",
            f"guardrail pass rate (floor {floor:.0%})",
            [f"{g.pass_rate:.0%}" for g in gates],
        ),
        ("hard gate", "round accepted", ["yes" if g.accepted else "no" for g in gates]),
        *(
            (
                "reported",
                label,
                [_reported_cell(m, attribute, pattern, key) for m in metrics],
            )
            for label, attribute, pattern, key in _REPORTED
        ),
    ]
    return pd.DataFrame(
        [
            {"kind": kind, "metric": name, **dict(zip(labels, cells, strict=True))}
            for kind, name, cells in rows
        ]
    )


def outcome_table(
    traces: Sequence[Trace], pages: Sequence[ReviewPage], variant: str
) -> pd.DataFrame:
    by_id = {t.doc_id: t for t in traces if t.variant == variant}
    missing = [p.doc_id for p in pages if p.doc_id not in by_id]
    if missing:
        raise ValueError(f"no trace for {missing[0]} in {variant}")
    table = page_table(pages)
    table["route"] = [by_id[p.doc_id].route for p in pages]
    table["reason"] = [by_id[p.doc_id].route_reason.replace("_", " ") for p in pages]
    return table


def follow_page(
    traces: Sequence[Trace],
    doc_id: str,
    variants: Sequence[str] = VARIANT_ORDER,
) -> pd.DataFrame:
    rows = []
    for variant in variants:
        own = [t for t in traces if t.variant == variant and t.doc_id == doc_id]
        if not own:
            raise ValueError(f"no trace for {doc_id} in {variant}")
        trace = own[0]
        models = ", ".join(dict.fromkeys(trace.models_used))
        outcome = ROUTE_LABELS[trace.route]
        if trace.attorney_decision is not None:
            outcome += f", then {trace.attorney_decision}"
        rows.append(
            {
                "round": trace.variant_label,
                "engine and model": f"{trace.engine}: {models}",
                "tools called": trace.tool_calls,
                "tokens": f"{trace.sync_tokens:,}",
                "seconds": f"{trace.latency_s:.2f}",
                "model cost": f"${trace.model_cost_usd:.5f}",
                "outcome": outcome,
                "reason": trace.route_reason.replace("_", " "),
            }
        )
    return pd.DataFrame(rows)


def confusion_table(
    traces: Sequence[Trace],
    variant: str,
    pages: Sequence[ReviewPage] = EVAL_SET,
) -> pd.DataFrame:
    table = pd.DataFrame(
        variant_metrics(traces, pages, variant).confusion,
        index=pd.Index(ROUTES, name="expected"),
        columns=pd.Index(ROUTES, name="actual"),
    )
    return table


def holdout_table(report: HoldoutReport) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy": s.policy,
            "split": s.split,
            "escalation rate": f"{s.score.escalation_rate:.0%}",
            "privileged leaks": s.score.privileged_leaks,
            "PII leaks": s.score.pii_leaks,
            "quality": f"{s.score.mean_quality:.2f}",
            "$/page": f"${s.score.blended_cost_per_page_usd:.3f}",
        }
        for s in report.scores
    )


__all__ = [
    "GateResult",
    "before_after",
    "confusion_table",
    "follow_page",
    "gate",
    "gate_table",
    "holdout_table",
    "metrics_table",
    "outcome_table",
    "page_table",
    "request_diff",
    "symptoms",
    "watch_percent",
]

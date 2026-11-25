"""One trace row per page per variant, the same schema for simulated and live runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from doc_review_opt.config import get_settings
from doc_review_opt.models import PageResult, ReviewPage, Route

SETTINGS = get_settings()
QUALITY = SETTINGS.quality
ATTORNEY_REVIEW_COST_USD_PER_PAGE = SETTINGS.attorney.review_cost_usd_per_page

Mode = Literal["simulated", "live"]


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BackgroundWork(_Row):
    tokens: int = 0
    cost_usd: float = 0.0


class Quality(_Row):
    privileged_leak: bool
    pii_leak: bool
    route_correct: bool
    escalation_correct: bool
    unnecessary_escalation: bool
    responsive_withheld: bool
    extra_tool_calls: int
    parse_failed: bool
    quality_score: float
    policy_compliant: bool


@dataclass(frozen=True)
class PageOutcome:
    route: Route
    decision: Literal["produce", "withhold"]
    escalated: bool
    produced_text: str | None
    tools: tuple[str, ...] = ()
    parse_failed: bool = False


@dataclass(frozen=True)
class RunInfo:
    variant: str
    label: str
    mode: Mode
    background: BackgroundWork = field(default_factory=BackgroundWork)


class Trace(_Row):
    variant: str
    variant_label: str
    mode: Mode
    doc_id: str
    difficulty: str
    risk: str
    engine: str
    models_used: tuple[str, ...]
    routing_tokens: int
    tool_calls: int
    expected_tools: tuple[str, ...]
    tools: tuple[str, ...]
    route: Route
    route_reason: str
    expected_route: Route
    decision: Literal["produce", "withhold"]
    expected_disposition: str
    escalated: bool
    must_escalate: bool
    attorney_decision: Literal["produce", "withhold"] | None
    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    output_tokens: int
    visible_output_tokens: int
    reasoning_tokens: int
    sync_tokens: int
    background_tokens: int
    total_tokens: int
    latency_s: float
    sync_cost_usd: float
    background_cost_usd: float
    model_cost_usd: float
    attorney_cost_usd: float
    cost_usd: float
    produced_text: str | None
    privileged_leak: bool
    pii_leak: bool
    route_correct: bool
    escalation_correct: bool
    unnecessary_escalation: bool
    responsive_withheld: bool
    extra_tool_calls: int
    parse_failed: bool
    quality_score: float
    policy_compliant: bool


def evaluate_trace_quality(page: ReviewPage, outcome: PageOutcome) -> Quality:
    labels = page.labels
    produced = outcome.decision == "produce"
    text = outcome.produced_text or ""
    privileged_leak = labels.privileged and produced
    pii_leak = any(span in text for span in labels.pii_spans)
    route_correct = outcome.route == labels.expected_route
    escalation_correct = outcome.escalated == labels.must_escalate
    unnecessary = outcome.escalated and not labels.must_escalate
    withheld = labels.expected_disposition == "produce" and not produced
    extra = len(set(outcome.tools) - set(labels.expected_tools))
    penalties = (
        (privileged_leak, QUALITY.leak_penalty),
        (pii_leak, QUALITY.leak_penalty),
        (not route_correct, QUALITY.wrong_route_penalty),
        (not escalation_correct, QUALITY.escalation_mismatch_penalty),
        (withheld, QUALITY.responsive_withheld_penalty),
        (unnecessary, QUALITY.unnecessary_escalation_penalty),
        (outcome.parse_failed, QUALITY.parse_failure_penalty),
    )
    score = QUALITY.base - sum(cost for hit, cost in penalties if hit)
    score -= min(QUALITY.extra_tool_penalty_cap, QUALITY.extra_tool_penalty * extra)
    return Quality(
        privileged_leak=privileged_leak,
        pii_leak=pii_leak,
        route_correct=route_correct,
        escalation_correct=escalation_correct,
        unnecessary_escalation=unnecessary,
        responsive_withheld=withheld,
        extra_tool_calls=extra,
        parse_failed=outcome.parse_failed,
        quality_score=round(max(0.0, min(QUALITY.max, score)), 2),
        policy_compliant=not (privileged_leak or pii_leak)
        and route_correct
        and escalation_correct,
    )


def trace_from_result(page: ReviewPage, result: PageResult, run: RunInfo) -> Trace:
    classification = result.classification
    if classification is None:
        raise ValueError(f"{result.doc_id}: result has no classification")
    failed = result.failed_redaction
    usage = classification.usage + failed.usage
    redaction = result.redaction
    if redaction is not None:
        usage = usage + redaction.usage
    sync_cost = (
        classification.cost_usd
        + failed.cost_usd
        + (redaction.cost_usd if redaction else 0.0)
    )
    model_cost = sync_cost + run.background.cost_usd
    attorney_cost = ATTORNEY_REVIEW_COST_USD_PER_PAGE if result.escalated else 0.0
    models = classification.models or (classification.model,)
    if redaction is not None and redaction.model not in models:
        models = (*models, redaction.model)
    sync_tokens = usage.input + usage.output
    outcome = PageOutcome(
        route=result.route,
        decision=result.decision,
        escalated=result.escalated,
        produced_text=result.produced_text,
        tools=classification.tools,
        parse_failed=classification.parse_failed,
    )
    quality = evaluate_trace_quality(page, outcome)
    labels = page.labels
    return Trace(
        variant=run.variant,
        variant_label=run.label,
        mode=run.mode,
        doc_id=page.doc_id,
        difficulty=labels.difficulty,
        risk=labels.risk,
        engine=classification.engine,
        models_used=models,
        routing_tokens=classification.routing_tokens,
        tool_calls=len(classification.tools),
        expected_tools=labels.expected_tools,
        tools=classification.tools,
        route=result.route,
        route_reason=result.reason,
        expected_route=labels.expected_route,
        decision=result.decision,
        expected_disposition=labels.expected_disposition,
        escalated=result.escalated,
        must_escalate=labels.must_escalate,
        attorney_decision=result.attorney_decision,
        input_tokens=usage.input,
        cached_tokens=usage.cached,
        cache_write_tokens=usage.cache_write,
        output_tokens=usage.output,
        visible_output_tokens=usage.output - usage.reasoning,
        reasoning_tokens=usage.reasoning,
        sync_tokens=sync_tokens,
        background_tokens=run.background.tokens,
        total_tokens=sync_tokens + run.background.tokens,
        latency_s=classification.latency_s + (redaction.latency_s if redaction else 0),
        sync_cost_usd=sync_cost,
        background_cost_usd=run.background.cost_usd,
        model_cost_usd=model_cost,
        attorney_cost_usd=attorney_cost,
        cost_usd=model_cost + attorney_cost,
        produced_text=result.produced_text,
        **quality.model_dump(),
    )


def deterministic_guardrail_check(
    page: ReviewPage,
    trace: Trace,
    *,
    interrupt_payload: Mapping[str, object] | None = None,
) -> list[str]:
    produced = trace.decision == "produce"
    checks = (
        ("privileged_page_produced", trace.privileged_leak),
        ("pii_in_produced_text", trace.pii_leak),
        ("responsive_page_withheld", trace.responsive_withheld),
        (
            "nonresponsive_page_produced",
            trace.expected_disposition == "withhold" and produced,
        ),
        ("escalation_mismatch", not trace.escalation_correct),
        ("unnecessary_attorney_review", trace.unnecessary_escalation),
        (
            "interrupt_payload_contains_text",
            _payload_has_text(page, interrupt_payload),
        ),
        ("unparseable_agent_output", trace.parse_failed),
        (
            "too_many_unnecessary_tools",
            trace.extra_tool_calls > QUALITY.max_extra_tools_before_flag,
        ),
        (
            "classifier_error_not_escalated",
            trace.route_reason == "classifier_error" and not trace.escalated,
        ),
    )
    return [code for code, hit in checks if hit]


def _payload_has_text(page: ReviewPage, payload: Mapping[str, object] | None) -> bool:
    if payload is None:
        return False
    dumped = json.dumps(payload, default=str)
    return page.text[: QUALITY.snippet_chars] in dumped or any(
        span in dumped for span in page.labels.pii_spans
    )


def traces_frame(traces: list[Trace]) -> pd.DataFrame:
    rows = []
    for trace in traces:
        row = trace.model_dump()
        row["models_used"] = ", ".join(trace.models_used)
        row["expected_tools"] = ", ".join(trace.expected_tools)
        row["tools"] = ", ".join(trace.tools)
        rows.append(row)
    return pd.DataFrame(rows)

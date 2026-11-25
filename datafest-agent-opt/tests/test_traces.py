"""Tests for the trace schema, quality score and guardrails."""

import pytest

from doc_review_opt.experiment.traces import (
    ATTORNEY_REVIEW_COST_USD_PER_PAGE,
    BackgroundWork,
    PageOutcome,
    RunInfo,
    Trace,
    deterministic_guardrail_check,
    evaluate_trace_quality,
    trace_from_result,
    traces_frame,
)
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.models import ClassifierResult, PageResult, ReviewPage, TokenUsage

PRIV = next(p for p in EVAL_SET if p.labels.privileged)
PII = next(p for p in EVAL_SET if p.labels.expected_route == "redact")
CLEAN = next(p for p in EVAL_SET if p.labels.expected_route == "produce")
RUN = RunInfo("v", "Variant", "simulated")


def outcome(page: ReviewPage, **fields: object) -> PageOutcome:
    base: dict[str, object] = {
        "route": page.labels.expected_route,
        "decision": "produce"
        if page.labels.expected_disposition == "produce"
        else "withhold",
        "escalated": page.labels.must_escalate,
        "produced_text": None,
    }
    return PageOutcome(**{**base, **fields})  # type: ignore[arg-type]


def test_perfect_outcome_scores_the_base() -> None:
    quality = evaluate_trace_quality(CLEAN, outcome(CLEAN, produced_text=CLEAN.text))
    assert quality.quality_score == 0.98
    assert quality.policy_compliant


def test_privileged_leak_costs_a_leak_and_a_route_penalty() -> None:
    bad = outcome(PRIV, route="produce", decision="produce", escalated=False)
    quality = evaluate_trace_quality(PRIV, bad)
    assert quality.privileged_leak
    assert quality.quality_score == pytest.approx(0.98 - 0.40 - 0.20 - 0.15, abs=0.01)
    assert not quality.policy_compliant


def test_pii_in_produced_text_is_a_leak() -> None:
    bad = outcome(PII, route="redact", decision="produce", produced_text=PII.text)
    quality = evaluate_trace_quality(PII, bad)
    assert quality.pii_leak
    assert quality.quality_score == pytest.approx(0.58)


def test_other_penalties_and_clamp() -> None:
    parse = outcome(CLEAN, produced_text=CLEAN.text, parse_failed=True)
    assert evaluate_trace_quality(CLEAN, parse).quality_score == pytest.approx(0.90)
    tools = outcome(
        CLEAN, produced_text=CLEAN.text, tools=("a", "b", "c", "d", "e", "f")
    )
    assert evaluate_trace_quality(CLEAN, tools).extra_tool_calls == 6
    assert evaluate_trace_quality(CLEAN, tools).quality_score == pytest.approx(0.84)
    withheld = outcome(CLEAN, decision="withhold", route="withhold")
    quality = evaluate_trace_quality(CLEAN, withheld)
    assert quality.responsive_withheld
    assert quality.quality_score == pytest.approx(0.98 - 0.20 - 0.10)
    over = outcome(
        CLEAN, escalated=True, route="attorney_review", produced_text=CLEAN.text
    )
    assert evaluate_trace_quality(CLEAN, over).unnecessary_escalation


def result_for(page: ReviewPage, *, escalated: bool = False) -> PageResult:
    classification = ClassifierResult(
        engine="agent",
        model="gpt-6.1-terra",
        usage=TokenUsage(input=1000, cached=200, output=100),
        latency_s=1.5,
        cost_usd=0.002,
        tools=("lookup_custodian",),
        models=("gpt-6.1-terra",),
        routing_tokens=7,
    )
    return PageResult(
        doc_id=page.doc_id,
        decision="produce",
        route="produce",
        produced_text=page.text,
        escalated=escalated,
        classification=classification,
        reason="clean_responsive",
    )


def test_trace_row_costs_and_tokens() -> None:
    run = RunInfo("v", "Variant", "live", BackgroundWork(tokens=50, cost_usd=0.001))
    trace = trace_from_result(CLEAN, result_for(CLEAN, escalated=True), run)
    assert trace.mode == "live"
    assert trace.sync_tokens == 1100
    assert trace.total_tokens == 1150
    assert trace.model_cost_usd == pytest.approx(0.003)
    assert trace.attorney_cost_usd == ATTORNEY_REVIEW_COST_USD_PER_PAGE
    assert trace.cost_usd == pytest.approx(0.003 + ATTORNEY_REVIEW_COST_USD_PER_PAGE)
    assert trace.route_reason == "clean_responsive"
    assert trace.extra_tool_calls == 1


def test_result_without_classification_is_rejected() -> None:
    bare = PageResult(doc_id="x", decision="withhold", route="withhold")
    with pytest.raises(ValueError, match="no classification"):
        trace_from_result(CLEAN, bare, RUN)


def test_guardrail_codes() -> None:
    leak = trace_from_result(CLEAN, result_for(CLEAN), RUN)
    forged = leak.model_copy(
        update={
            "privileged_leak": True,
            "pii_leak": True,
            "responsive_withheld": True,
            "expected_disposition": "withhold",
            "escalation_correct": False,
            "unnecessary_escalation": True,
            "parse_failed": True,
            "extra_tool_calls": 3,
            "route_reason": "classifier_error",
        }
    )
    codes = deterministic_guardrail_check(CLEAN, forged)
    assert codes == [
        "privileged_page_produced",
        "pii_in_produced_text",
        "responsive_page_withheld",
        "nonresponsive_page_produced",
        "escalation_mismatch",
        "unnecessary_attorney_review",
        "unparseable_agent_output",
        "too_many_unnecessary_tools",
        "classifier_error_not_escalated",
    ]
    assert deterministic_guardrail_check(CLEAN, leak) == []


def test_interrupt_payload_with_page_text_is_flagged() -> None:
    trace = trace_from_result(CLEAN, result_for(CLEAN), RUN)
    clean = {"doc_id": CLEAN.doc_id, "privileged": 0.1}
    dirty = {"doc_id": CLEAN.doc_id, "excerpt": CLEAN.text}
    assert deterministic_guardrail_check(CLEAN, trace, interrupt_payload=clean) == []
    assert deterministic_guardrail_check(CLEAN, trace, interrupt_payload=dirty) == [
        "interrupt_payload_contains_text"
    ]


def test_traces_frame_joins_tuple_columns() -> None:
    trace = trace_from_result(CLEAN, result_for(CLEAN), RUN)
    frame = traces_frame([trace])
    assert frame.loc[0, "tools"] == "lookup_custodian"
    assert isinstance(trace, Trace)
    assert len(frame) == 1

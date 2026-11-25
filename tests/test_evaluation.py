"""Tests for deterministic guardrails and the optional judge."""

import json
from typing import Any, Literal

import httpx2
import pytest

from doc_review_opt.experiment.evaluation import (
    evaluate_answer_traces,
    summarize_answer_evals,
)
from doc_review_opt.experiment.live import BudgetTracker
from doc_review_opt.experiment.traces import RunInfo, trace_from_result
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.models import ClassifierResult, PageResult
from tests.helpers import at
from tests.test_decisions_classifier import client_for
from tests.test_live_api import text_response

PAGES = EVAL_SET[:2]
LEAKED = next(p for p in EVAL_SET if p.labels.privileged)


def trace_for(
    page: Any, *, decision: Literal["produce", "withhold"] = "produce"
) -> Any:
    result = PageResult(
        doc_id=page.doc_id,
        decision=decision,
        route="produce" if decision == "produce" else "withhold",
        produced_text=page.text if decision == "produce" else None,
        classification=ClassifierResult(engine="agent", model="gpt-6.1-terra"),
    )
    return trace_from_result(page, result, RunInfo("v", "Variant", "simulated"))


async def test_without_a_client_every_row_is_not_run() -> None:
    frame = await evaluate_answer_traces(PAGES, [trace_for(p) for p in PAGES])
    assert set(frame["judge_status"]) == {"not_run"}
    summary = summarize_answer_evals(frame)
    assert at(summary, 0, "judge_coverage") == 0
    assert at(summary, 0, "judge_graded") == 0
    assert at(summary, 0, "deterministic_pass_rate") == 1.0


async def test_judge_pass_never_overrides_a_deterministic_failure() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        verdict = {"passed": True, "reason": "looks fine"}
        return httpx2.Response(200, json=text_response(json.dumps(verdict)))

    pages = [PAGES[0], LEAKED]
    traces = [trace_for(PAGES[0]), trace_for(LEAKED)]
    frame = await evaluate_answer_traces(
        pages, traces, client=client_for(handler), budget=BudgetTracker()
    )
    summary = summarize_answer_evals(frame).iloc[0]
    assert summary["judge_pass_rate"] == 1.0
    assert summary["deterministic_pass_rate"] == 0.5
    assert summary["both_pass_rate"] == 0.5
    assert summary["judge_coverage"] == 1.0
    assert summary["known_judge_cost_usd"] > 0
    assert "privileged_page_produced" in at(frame, 1, "deterministic_failures")


async def test_invalid_judge_json_and_request_errors_are_errors() -> None:
    replies = iter(
        [httpx2.Response(200, json=text_response("not json")), httpx2.Response(500)]
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        return next(replies)

    frame = await evaluate_answer_traces(
        PAGES,
        [trace_for(p) for p in PAGES],
        client=client_for(handler),
        budget=BudgetTracker(),
    )
    assert sorted(frame["judge_status"]) == ["error", "error"]
    summary = summarize_answer_evals(frame).iloc[0]
    assert summary["judge_errors"] == 2
    assert pytest.approx(0.0) == summary["judge_coverage"]


async def test_judge_stops_issuing_calls_once_the_budget_is_reached() -> None:
    calls = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        verdict = {"passed": True, "reason": "fine"}
        return httpx2.Response(200, json=text_response(json.dumps(verdict)))

    pages = EVAL_SET[:10]
    budget = BudgetTracker(limit_usd=0.045)
    frame = await evaluate_answer_traces(
        pages,
        [trace_for(p) for p in pages],
        client=client_for(handler),
        budget=budget,
    )
    refused = frame[frame["reason"].str.contains("Budget reached")]
    assert 0 < len(calls) < len(pages)
    assert len(refused) == len(pages) - len(calls)
    assert set(refused["judge_status"]) == {"error"}
    assert budget.total_usd > 0


async def test_judge_client_without_a_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="budget"):
        await evaluate_answer_traces(
            PAGES, [trace_for(p) for p in PAGES], client=client_for(lambda r: r)
        )

"""Tests that failed live calls still report, and the budget counts, their spend."""

import json
import logging
from typing import Any

import httpx2
import pytest

from doc_review_opt.agent.classifiers.decisions import DecisionsClassifier
from doc_review_opt.agent.classifiers.llm_probability import LLMProbabilityClassifier
from doc_review_opt.agent.classifiers.tool_agent import LiveAgentClassifier
from doc_review_opt.agent.graph import (
    build_review_graph,
    review_checkpointer,
    run_batch,
)
from doc_review_opt.agent.redactor import OpenAIRedactor, RedactionError
from doc_review_opt.agent.variants import CONTROLS, ROUTING
from doc_review_opt.experiment.live import (
    BudgetedClassifier,
    BudgetedRedactor,
    BudgetTracker,
)
from doc_review_opt.experiment.traces import RunInfo, trace_from_result
from doc_review_opt.fixtures import PROTOCOL
from tests.helpers import PAGE, StubClassifier, make_result
from tests.test_decisions_classifier import (
    OK_ANSWERS,
    client_for,
    decisions_body,
)
from tests.test_live_api import PAGE as TOOL_PAGE
from tests.test_live_api import Script, call_response, text_response

TRIAGE = {
    "responsive": True,
    "privileged": True,
    "pii": False,
    "needs_attorney": False,
}


async def test_decisions_reply_that_fails_parsing_still_bills() -> None:
    answers = [OK_ANSWERS[0], OK_ANSWERS[2], OK_ANSWERS[1]]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=decisions_body(answers))

    budget = BudgetTracker()
    classifier = BudgetedClassifier(DecisionsClassifier(client_for(handler)), budget)
    result = await classifier.classify(PAGE, PROTOCOL)
    assert result.error is not None
    assert result.usage.input == 2000
    assert result.cost_usd == pytest.approx((500 * 0.10 + 1500 * 0.01) / 1e6)
    assert budget.total_usd == pytest.approx(result.cost_usd)


@pytest.mark.parametrize("status", ["completed", "incomplete"])
async def test_llm_second_opinion_failure_still_bills(status: str) -> None:
    text = "not json" if status == "completed" else ""
    script = Script([text_response(text, status=status)])
    budget = BudgetTracker()
    classifier = LLMProbabilityClassifier(client_for(script), "gpt-6.1-terra")
    result = await BudgetedClassifier(classifier, budget).classify(PAGE, PROTOCOL)
    assert result.error is not None
    assert result.usage.input == 3000
    assert result.cost_usd == pytest.approx((3000 * 1.00 + 60 * 5.00) / 1e6)
    assert budget.total_usd == pytest.approx(result.cost_usd)


async def test_tool_loop_failure_after_successful_triage_bills_both() -> None:
    script = Script(
        [text_response(json.dumps(TRIAGE)), text_response("", status="incomplete")]
    )
    budget = BudgetTracker()
    agent = LiveAgentClassifier(client_for(script), ROUTING)
    result = await BudgetedClassifier(agent, budget).classify(PAGE, PROTOCOL)
    assert result.error is not None
    assert result.usage.input == 6000
    assert result.model == "gpt-6.1-terra"
    triage_cost = (3000 * 0.10 + 60 * 0.50) / 1e6
    loop_cost = (3000 * 1.00 + 60 * 5.00) / 1e6
    assert result.cost_usd == pytest.approx(triage_cost + loop_cost)
    assert budget.total_usd == pytest.approx(0.00363)


async def test_tool_loop_with_leftover_calls_bills_every_call() -> None:
    script = Script([call_response("lookup_custodian", {"name": "x"})] * 4)
    agent = LiveAgentClassifier(client_for(script), CONTROLS)
    result = await agent.classify(TOOL_PAGE, PROTOCOL)
    assert result.error is not None
    assert result.usage.input == 12000
    assert result.cost_usd == pytest.approx((12000 * 1.00 + 240 * 5.00) / 1e6)


async def test_tool_loop_api_error_after_billed_calls_keeps_their_cost() -> None:
    replies = [call_response("lookup_custodian", {"name": "x"})]

    def handler(request: httpx2.Request) -> httpx2.Response:
        if replies:
            return httpx2.Response(200, json=replies.pop())
        return httpx2.Response(500, json={"error": {"message": "down"}})

    agent = LiveAgentClassifier(client_for(handler), CONTROLS)
    result = await agent.classify(TOOL_PAGE, PROTOCOL)
    assert result.error is not None
    assert result.usage.input == 3000
    assert result.cost_usd == pytest.approx((3000 * 1.00 + 60 * 5.00) / 1e6)


async def test_failed_triage_still_bills() -> None:
    script = Script([text_response("not json")])
    result = await LiveAgentClassifier(client_for(script), ROUTING).classify(
        PAGE, PROTOCOL
    )
    assert result.error is not None
    assert result.usage.input == 3000
    assert result.cost_usd == pytest.approx((3000 * 0.10 + 60 * 0.50) / 1e6)


async def test_incomplete_redaction_is_charged_to_the_budget() -> None:
    script: Any = Script([text_response("", status="incomplete")])
    budget = BudgetTracker()
    redactor = BudgetedRedactor(OpenAIRedactor(client_for(script)), budget)
    with pytest.raises(RedactionError) as raised:
        await redactor.redact(PAGE)
    assert raised.value.usage.input == 3000
    assert raised.value.cost_usd == pytest.approx((3000 * 0.10 + 60 * 0.50) / 1e6)
    assert budget.total_usd == pytest.approx(0.00033)


async def test_failed_redactions_reach_the_trace_and_match_the_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    script = Script([text_response("", status="incomplete")] * 2)
    budget = BudgetTracker()
    redactor = BudgetedRedactor(OpenAIRedactor(client_for(script)), budget)
    stub = StubClassifier(
        make_result(responsive=2.0, confidence=0.95, privileged=0.0, pii=0.99)
    )
    graph = build_review_graph(stub, redactor, review_checkpointer())
    with caplog.at_level(logging.WARNING):
        outcome = await run_batch(
            graph,
            [PAGE],
            thread_id="t1",
            attorney_decisions={PAGE.doc_id: "produce"},
        )
    (result,) = outcome.results
    trace = trace_from_result(PAGE, result, RunInfo("v", "V", "live"))
    assert (result.decision, len(script.requests)) == ("withhold", 2)
    assert budget.total_usd > 0
    assert trace.model_cost_usd == pytest.approx(budget.total_usd)
    assert trace.input_tokens == 6000
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "status=incomplete" in logged
    assert PAGE.text not in logged

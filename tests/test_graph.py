"""Tests for the review graph, interrupts and redaction."""

import json
import logging
import warnings
from typing import Any

import httpx2
import pytest
from langgraph.types import Command

from doc_review_opt.agent.classifiers.decisions import DecisionsClassifier
from doc_review_opt.agent.graph import (
    BatchOutcome,
    ReviewGraph,
    build_review_graph,
    review_checkpointer,
    run_batch,
)
from doc_review_opt.agent.nodes import InvalidAttorneyDecisionError
from doc_review_opt.agent.redactor import OpenAIRedactor
from doc_review_opt.experiment.simulation import SimulatedRedactor
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.models import ClassifierResult, ReviewPage
from tests.helpers import (
    PAGE,
    REDACTED_TEXT,
    FakeRedactor,
    StubClassifier,
    make_result,
    refused,
)
from tests.test_decisions_classifier import (
    OK_ANSWERS,
    client_for,
    decisions_body,
    responses_body,
)

CONFIG: Any = {"configurable": {"thread_id": "t1"}}
CLEAN = make_result(responsive=2.0, confidence=0.95, privileged=0.01, pii=0.02)
PRIVILEGED = make_result(responsive=2.0, confidence=0.99, privileged=0.9, pii=0.0)


def graph_for(classifier: Any, redactor: Any = None) -> ReviewGraph:
    return build_review_graph(
        classifier, redactor or FakeRedactor(), review_checkpointer()
    )


async def first_pass(graph: ReviewGraph, pages: list[ReviewPage]) -> dict[str, Any]:
    return await graph.ainvoke({"pages": pages, "results": []}, CONFIG)


async def test_clean_responsive_page_is_produced() -> None:
    state = await first_pass(graph_for(StubClassifier(CLEAN)), [PAGE])
    (result,) = state["results"]
    assert (result.doc_id, result.decision, result.route) == (
        PAGE.doc_id,
        "produce",
        "produce",
    )
    assert result.produced_text == PAGE.text
    assert result.classification is not None


async def test_non_responsive_page_is_withheld() -> None:
    stub = StubClassifier(
        make_result(responsive=0.1, confidence=0.95, privileged=0.0, pii=0.0)
    )
    state = await first_pass(graph_for(stub), [PAGE])
    assert state["results"][0].decision == "withhold"
    assert state["results"][0].produced_text is None


async def test_fractional_score_just_under_threshold_is_withheld() -> None:
    stub = StubClassifier(
        make_result(responsive=0.49, confidence=0.95, privileged=0.0, pii=0.0)
    )
    state = await first_pass(graph_for(stub), [PAGE])
    assert state["results"][0].decision == "withhold"


async def test_pii_page_is_redacted_before_production() -> None:
    stub = StubClassifier(
        make_result(responsive=2.0, confidence=0.95, privileged=0.0, pii=0.91)
    )
    state = await first_pass(graph_for(stub), [PAGE])
    (result,) = state["results"]
    assert (result.decision, result.route, result.redacted) == (
        "produce",
        "redact",
        True,
    )
    assert result.produced_text == REDACTED_TEXT
    assert result.redaction is not None


async def test_redaction_output_replaces_the_original_page_text() -> None:
    secret = next(p for p in EVAL_SET if p.labels.pii_spans)
    stub = StubClassifier(
        make_result(responsive=2.0, confidence=0.95, privileged=0.0, pii=0.99)
    )
    state = await first_pass(graph_for(stub), [secret])
    assert all(
        span not in state["results"][0].produced_text
        for span in secret.labels.pii_spans
    )


@pytest.mark.parametrize(
    ("privileged", "confidence"),
    [(0.25, 0.95), (0.01, 0.5)],
    ids=["privilege_above_threshold", "low_responsive_confidence"],
)
async def test_uncertain_pages_escalate_to_attorney_review(
    privileged: float, confidence: float
) -> None:
    stub = StubClassifier(
        make_result(
            responsive=2.0, confidence=confidence, privileged=privileged, pii=0.0
        )
    )
    state = await first_pass(graph_for(stub), [PAGE])
    assert state["results"] == []
    assert state["__interrupt__"]


async def test_privilege_escalation_wins_over_pii_redaction() -> None:
    stub = StubClassifier(
        make_result(responsive=2.0, confidence=0.99, privileged=0.9, pii=0.99)
    )
    state = await first_pass(graph_for(stub), [PAGE])
    assert state["results"] == []
    assert state["__interrupt__"]


async def test_ambiguous_middle_band_escalates_despite_high_confidence() -> None:
    stub = StubClassifier(
        make_result(responsive=1.2, confidence=0.98, privileged=0.0, pii=0.0)
    )
    state = await first_pass(graph_for(stub), [PAGE])
    assert state["results"] == []
    assert state["__interrupt__"]


async def test_decisive_score_survives_mediocre_confidence() -> None:
    stub = StubClassifier(
        make_result(responsive=0.07, confidence=0.55, privileged=0.0, pii=0.0)
    )
    state = await first_pass(graph_for(stub), [PAGE])
    assert state["results"][0].decision == "withhold"


async def test_refusal_escalates() -> None:
    result = CLEAN.model_copy(
        update={"answers": {**CLEAN.answers, "privileged": refused("privileged")}}
    )
    state = await first_pass(graph_for(StubClassifier(result)), [PAGE])
    assert state["__interrupt__"][0].value["reason"] == "missing_or_refused_answer"


async def test_interrupt_payload_excludes_page_text() -> None:
    state = await first_pass(graph_for(StubClassifier(PRIVILEGED)), [PAGE])
    payload = state["__interrupt__"][0].value
    assert PAGE.text not in json.dumps(payload)
    assert payload["doc_id"] == PAGE.doc_id
    assert payload["privileged"] == 0.9
    assert set(payload) == {
        "doc_id",
        "reason",
        "privileged",
        "responsive_score",
        "responsive_confidence",
    }


async def test_attorney_decision_resumes_and_records_result() -> None:
    graph = graph_for(StubClassifier(PRIVILEGED))
    state = await first_pass(graph, [PAGE])
    resume = {state["__interrupt__"][0].id: "withhold"}
    final = await graph.ainvoke(Command(resume=resume), CONFIG)
    (result,) = final["results"]
    assert (result.decision, result.escalated, result.attorney_decision) == (
        "withhold",
        True,
        "withhold",
    )


async def test_invalid_attorney_decision_is_rejected() -> None:
    graph = graph_for(StubClassifier(PRIVILEGED))
    state = await first_pass(graph, [PAGE])
    resume = {state["__interrupt__"][0].id: "attorney_review"}
    with pytest.raises(InvalidAttorneyDecisionError):
        await graph.ainvoke(Command(resume=resume), CONFIG)


async def test_batch_fans_out_and_resumes_only_escalated_pages() -> None:
    other = next(p for p in EVAL_SET if p.doc_id != PAGE.doc_id)

    def by_page(page: ReviewPage) -> ClassifierResult:
        return PRIVILEGED if page.doc_id == other.doc_id else CLEAN

    stub = StubClassifier(by_page)
    graph = graph_for(stub)
    state = await first_pass(graph, [PAGE, other])
    assert [r.doc_id for r in state["results"]] == [PAGE.doc_id]
    assert len(state["__interrupt__"]) == 1

    resume = {state["__interrupt__"][0].id: "produce"}
    final = await graph.ainvoke(Command(resume=resume), CONFIG)
    assert {r.doc_id: r.decision for r in final["results"]} == {
        PAGE.doc_id: "produce",
        other.doc_id: "produce",
    }
    assert sorted(stub.seen) == sorted([PAGE.doc_id, other.doc_id])


async def test_run_batch_resumes_with_scripted_decisions_and_default() -> None:
    pages = list(EVAL_SET[:3])
    graph = graph_for(StubClassifier(PRIVILEGED))
    outcome = await run_batch(
        graph,
        pages,
        thread_id="demo",
        attorney_decisions={pages[0].doc_id: "produce"},
    )
    assert isinstance(outcome, BatchOutcome)
    decisions = {r.doc_id: r.decision for r in outcome.results}
    assert decisions == {
        pages[0].doc_id: "produce",
        pages[1].doc_id: "withhold",
        pages[2].doc_id: "withhold",
    }
    assert len(outcome.interrupts) == 3
    assert all(p.text not in json.dumps(outcome.interrupts) for p in pages)


async def test_graph_drives_a_real_decisions_classifier_over_mock_transport() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=decisions_body(OK_ANSWERS))

    graph = graph_for(DecisionsClassifier(client_for(handler)))
    state = await first_pass(graph, [PAGE])
    (result,) = state["results"]
    assert result.decision == "produce"
    assert result.classification is not None
    assert result.classification.engine == "decisions"
    assert result.classification.cost_usd > 0


async def test_openai_redactor_over_mock_transport() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        body = responses_body({})
        body["output"][0]["content"][0]["text"] = REDACTED_TEXT
        return httpx2.Response(200, json=body)

    redactor = OpenAIRedactor(client_for(handler))
    result = await redactor.redact(PAGE)
    assert result.text == REDACTED_TEXT
    assert captured["body"]["model"] == "gpt-6-luna"
    assert captured["body"]["input"].endswith(PAGE.text)
    assert result.cost_usd > 0


async def test_failed_redaction_escalates_instead_of_producing() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"error": {"message": "down"}})

    stub = StubClassifier(
        make_result(responsive=2.0, confidence=0.95, privileged=0.0, pii=0.99)
    )
    graph = graph_for(stub, OpenAIRedactor(client_for(handler)))
    state = await first_pass(graph, [PAGE])
    assert state["results"] == []
    assert state["__interrupt__"][0].value["reason"] == "redaction_failed"


async def resume_with(
    graph: ReviewGraph, pages: list[ReviewPage], decision: str
) -> Any:
    state = await first_pass(graph, pages)
    resume = {i.id: decision for i in state["__interrupt__"]}
    return await graph.ainvoke(Command(resume=resume), CONFIG)


def pii_signals_that_cannot_be_trusted() -> list[ClassifierResult]:
    typed = make_result(responsive=1.0, confidence=0.9, privileged=0.0, pii=0.0)
    return [
        ClassifierResult(engine="stub", model="m", error="TimeoutError: slow"),
        typed.model_copy(
            update={
                "answers": {**typed.answers, "contains_pii": refused("contains_pii")}
            }
        ),
        ClassifierResult(
            engine="agent",
            model="m",
            agent_disposition="attorney_review",
            agent_redact=True,
        ),
        ClassifierResult(
            engine="agent", model="m", agent_disposition="attorney_review"
        ),
    ]


@pytest.mark.parametrize("result", pii_signals_that_cannot_be_trusted())
async def test_attorney_produce_redacts_when_the_pii_signal_is_unusable(
    result: ClassifierResult,
) -> None:
    secret = next(p for p in EVAL_SET if p.labels.pii_spans)
    graph = graph_for(StubClassifier(result), SimulatedRedactor())
    state = await resume_with(graph, [secret], "produce")
    (outcome,) = state["results"]
    assert outcome.decision == "produce"
    assert outcome.redacted
    assert outcome.produced_text is not None
    assert not [
        span for span in secret.labels.pii_spans if span in outcome.produced_text
    ]


async def test_agent_that_said_produce_without_redaction_keeps_the_page() -> None:
    result = ClassifierResult(
        engine="agent",
        model="m",
        agent_disposition="attorney_review",
        agent_redact=False,
    )
    state = await resume_with(graph_for(StubClassifier(result)), [PAGE], "produce")
    assert not state["results"][0].redacted


async def test_attorney_produce_on_pii_page_is_redacted_first() -> None:
    secret = next(p for p in EVAL_SET if p.labels.pii_spans)
    stub = StubClassifier(
        make_result(responsive=1.0, confidence=0.9, privileged=0.0, pii=0.95)
    )
    state = await resume_with(graph_for(stub), [secret], "produce")
    (result,) = state["results"]
    assert (result.decision, result.route, result.redacted) == (
        "produce",
        "attorney_review",
        True,
    )
    assert result.produced_text == REDACTED_TEXT
    assert result.redaction is not None


async def test_attorney_produce_without_pii_keeps_original_text() -> None:
    stub = StubClassifier(
        make_result(responsive=1.0, confidence=0.9, privileged=0.0, pii=0.01)
    )
    state = await resume_with(graph_for(stub), [PAGE], "produce")
    assert state["results"][0].produced_text == PAGE.text
    assert not state["results"][0].redacted


async def test_attorney_produce_on_agent_redact_flag_is_redacted() -> None:
    stub = StubClassifier(
        make_result(responsive=2.0, confidence=0.9, privileged=0.9, pii=0.0).model_copy(
            update={"agent_disposition": "redact"}
        )
    )
    state = await resume_with(graph_for(stub), [PAGE], "produce")
    assert state["results"][0].redacted


async def test_attorney_produce_withholds_when_redaction_fails() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"error": {"message": "down"}})

    secret = next(p for p in EVAL_SET if p.labels.pii_spans)
    stub = StubClassifier(
        make_result(responsive=1.0, confidence=0.9, privileged=0.0, pii=0.95)
    )
    graph = graph_for(stub, OpenAIRedactor(client_for(handler)))
    state = await resume_with(graph, [secret], "produce")
    (result,) = state["results"]
    assert (result.decision, result.produced_text) == ("withhold", None)


async def test_interrupt_and_resume_emit_no_deserialization_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph = graph_for(StubClassifier(PRIVILEGED))
    with caplog.at_level(logging.DEBUG), warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        state = await first_pass(graph, [PAGE])
        resume = {state["__interrupt__"][0].id: "produce"}
        final = await graph.ainvoke(Command(resume=resume), CONFIG)
    assert final["results"][0].escalated
    noise = [r.getMessage() for r in caplog.records] + [str(w.message) for w in seen]
    assert not [m for m in noise if "unregistered type" in m]

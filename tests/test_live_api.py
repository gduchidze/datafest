"""Tests for live reviewers, cascade, budget, Batch rows and the judge."""

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from doc_review_opt.agent.classifiers.cascade import (
    CascadeClassifier,
    PriorProductionGate,
    in_ambiguous_band,
)
from doc_review_opt.agent.classifiers.dispositions import (
    parse_free_text,
    parse_json_disposition,
)
from doc_review_opt.agent.classifiers.tool_agent import (
    LiveAgentClassifier,
    live_config_for_page,
    run_tool_loop,
)
from doc_review_opt.agent.routing import DEFAULT_POLICY, decide
from doc_review_opt.agent.variants import BASELINE, CONTROLS, ROUTING
from doc_review_opt.config import get_settings
from doc_review_opt.exceptions import BilledCallError, BudgetExceededError
from doc_review_opt.experiment.batch import (
    background_work,
    build_background_batch,
    write_batch_jsonl,
)
from doc_review_opt.experiment.live import (
    JUDGE_MODEL,
    BudgetedClassifier,
    BudgetTracker,
    _trace_summary,
    build_live_engines,
    judge_page,
    run_live_variant,
)
from doc_review_opt.experiment.traces import RunInfo, trace_from_result
from doc_review_opt.fixtures import EVAL_SET, PROTOCOL
from doc_review_opt.models import ClassifierResult, PredicateAnswer
from tests.helpers import StubClassifier, make_result
from tests.test_decisions_classifier import (
    OK_ANSWERS,
    client_for,
    decisions_body,
    responses_body,
    score_answer,
)

LIVE = get_settings().live
RUN_PAGE_RESERVE_USD = LIVE.per_page_ceiling_usd + LIVE.redaction_call_ceiling_usd
PAGE = next(
    p
    for p in EVAL_SET
    if p.labels.expected_tools == ("lookup_custodian", "escalate_to_attorney")
)
JSON_REPLY = {
    "disposition": "attorney_review",
    "redact": False,
    "reason": "attorney named",
}


def text_response(text: str, *, status: str = "completed") -> dict[str, Any]:
    body = responses_body({})
    body["output"][0]["content"][0]["text"] = text
    body["status"] = status
    if status != "completed":
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    return body


def call_response(
    name: str, arguments: dict[str, Any], call_id: str = "call_1"
) -> dict[str, Any]:
    body = text_response("")
    body["output"] = [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments),
            "status": "completed",
        }
    ]
    return body


class Script:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return httpx2.Response(200, json=self.replies[len(self.requests) - 1])


async def test_tool_loop_forwards_output_and_sets_none_on_the_last_round() -> None:
    script = Script(
        [
            call_response("lookup_custodian", {"name": "Priya Raghunathan"}),
            call_response(
                "escalate_to_attorney", {"doc_id": PAGE.doc_id, "reason": "r"}, "call_2"
            ),
            text_response(json.dumps(JSON_REPLY)),
        ]
    )
    config = live_config_for_page(PAGE, CONTROLS)
    loop = await run_tool_loop(client_for(script), config, PAGE, max_tool_rounds=2)
    assert loop.tools == ("lookup_custodian", "escalate_to_attorney")
    first, second, third = script.requests
    assert first["tool_choice"]["type"] == "allowed_tools"
    assert first["reasoning"] == {"effort": "none"}
    outputs = [i for i in second["input"] if i.get("type") == "function_call_output"]
    assert outputs[0]["call_id"] == "call_1"
    assert json.loads(outputs[0]["output"])["found"] is True
    assert any(i.get("type") == "function_call" for i in second["input"])
    assert second["tool_choice"] != "none"
    assert third["tool_choice"] == "none"
    assert loop.usage.input == 9000


async def test_tool_loop_closes_early_when_the_page_ceiling_is_near() -> None:
    script = Script(
        [
            call_response("lookup_custodian", {"name": "Priya Raghunathan"}),
            text_response(json.dumps(JSON_REPLY)),
        ]
    )
    config = live_config_for_page(PAGE, CONTROLS)
    loop = await run_tool_loop(client_for(script), config, PAGE, ceiling_usd=0.012)
    assert [r["tool_choice"] == "none" for r in script.requests] == [False, True]
    assert loop.usage.input == 6000
    assert loop.usage.output == 120
    assert (6000 * 1.0 + 120 * 5.0) / 1e6 <= 0.012


async def test_tool_loop_stops_when_the_ceiling_leaves_no_room_to_close() -> None:
    script = Script([call_response("lookup_custodian", {"name": "x"})] * 2)
    config = live_config_for_page(PAGE, CONTROLS)
    with pytest.raises(BilledCallError, match="ceiling") as raised:
        await run_tool_loop(client_for(script), config, PAGE, ceiling_usd=0.008)
    assert len(script.requests) == 1
    assert raised.value.cost_usd == pytest.approx((3000 * 1.0 + 60 * 5.0) / 1e6)


async def test_incomplete_response_becomes_an_error_result() -> None:
    script = Script([text_response("", status="incomplete")])
    result = await LiveAgentClassifier(client_for(script), CONTROLS).classify(
        PAGE, PROTOCOL
    )
    assert result.error is not None
    assert "did not complete" in result.error


async def test_leftover_tool_calls_become_an_error_result() -> None:
    script = Script([call_response("lookup_custodian", {"name": "x"})] * 4)
    result = await LiveAgentClassifier(client_for(script), CONTROLS).classify(
        PAGE, PROTOCOL
    )
    assert result.error is not None
    assert "unexecuted tool calls" in result.error


async def test_strict_json_reply_becomes_a_disposition() -> None:
    script = Script([text_response(json.dumps(JSON_REPLY))])
    result = await LiveAgentClassifier(client_for(script), CONTROLS).classify(
        PAGE, PROTOCOL
    )
    assert result.agent_disposition == "attorney_review"
    assert not result.parse_failed


async def test_baseline_free_text_is_parsed_with_a_regex() -> None:
    reply = "Thorough review done. FINAL DISPOSITION: produce; redact: yes"
    script = Script([text_response(reply)])
    result = await LiveAgentClassifier(client_for(script), BASELINE).classify(
        PAGE, PROTOCOL
    )
    assert result.agent_disposition == "redact"
    assert script.requests[0]["tool_choice"] == "required"
    assert "format" not in script.requests[0]["text"]


async def test_unparseable_free_text_goes_to_an_attorney() -> None:
    script = Script([text_response("I looked at it and it seems fine.")])
    result = await LiveAgentClassifier(client_for(script), BASELINE).classify(
        PAGE, PROTOCOL
    )
    assert result.agent_disposition == "attorney_review"
    assert result.parse_failed


def triage_reply(**flags: bool) -> dict[str, Any]:
    base = {
        "responsive": True,
        "privileged": False,
        "pii": False,
        "needs_attorney": False,
    }
    return text_response(json.dumps({**base, **flags}))


@pytest.mark.parametrize(
    ("flags", "model"),
    [
        ({}, "gpt-6-luna"),
        ({"pii": True}, "gpt-6-luna"),
        ({"privileged": True}, "gpt-6.1-terra"),
    ],
)
async def test_triage_picks_luna_or_terra_by_risk(
    flags: dict[str, bool], model: str
) -> None:
    script = Script([triage_reply(**flags), text_response(json.dumps(JSON_REPLY))])
    result = await LiveAgentClassifier(client_for(script), ROUTING).classify(
        PAGE, PROTOCOL
    )
    assert script.requests[0]["model"] == "gpt-6-luna"
    assert script.requests[1]["model"] == model
    assert result.models == ("gpt-6-luna", model)
    assert result.routing_tokens > 0


def test_agent_variants_only_for_agent_engine() -> None:
    with pytest.raises(ValueError, match="not an agent variant"):
        live_config_for_page(PAGE, "05_split_workflow")
    with pytest.raises(ValueError, match="unknown variant"):
        build_live_engines("nope", client_for(lambda r: httpx2.Response(500)))


def band_result(**kwargs: float) -> ClassifierResult:
    values = {"responsive": 2.0, "confidence": 0.9, "privileged": 0.0, "pii": 0.0}
    return make_result(**{**values, **kwargs})


async def test_cascade_only_calls_the_second_opinion_in_the_ambiguous_band() -> None:
    clear = StubClassifier(band_result())
    ambiguous = StubClassifier(band_result(responsive=1.0))
    second = StubClassifier(band_result())
    page = EVAL_SET[0]
    await CascadeClassifier(clear, second, DEFAULT_POLICY).classify(page, "p")
    assert second.seen == []
    merged = await CascadeClassifier(ambiguous, second, DEFAULT_POLICY).classify(
        page, "p"
    )
    assert second.seen == [page.doc_id]
    assert merged.engine == "cascade"
    assert merged.answers["responsive"].score == 2.0  # type: ignore[union-attr]


def test_band_is_responsiveness_only() -> None:
    assert in_ambiguous_band(band_result(responsive=1.0), DEFAULT_POLICY)
    assert in_ambiguous_band(band_result(confidence=0.3), DEFAULT_POLICY)
    assert not in_ambiguous_band(band_result(privileged=0.3), DEFAULT_POLICY)
    assert not in_ambiguous_band(band_result(privileged=0.9), DEFAULT_POLICY)
    assert not in_ambiguous_band(band_result(), DEFAULT_POLICY)
    failed = ClassifierResult(engine="x", model="m", error="boom")
    assert not in_ambiguous_band(failed, DEFAULT_POLICY)


async def test_second_opinion_cannot_clear_a_privilege_flag() -> None:
    primary = StubClassifier(band_result(privileged=0.35))
    second = StubClassifier(band_result(privileged=0.05))
    merged = await CascadeClassifier(primary, second, DEFAULT_POLICY).classify(
        EVAL_SET[0], "p"
    )
    assert second.seen == []
    assert decide(merged, DEFAULT_POLICY).route == "attorney_review"


async def test_second_opinion_cannot_clear_a_pii_flag() -> None:
    primary = StubClassifier(band_result(responsive=1.0, pii=0.9))
    second = StubClassifier(band_result(responsive=2.0, confidence=0.9, pii=0.1))
    merged = await CascadeClassifier(primary, second, DEFAULT_POLICY).classify(
        EVAL_SET[0], "p"
    )
    assert second.seen == [EVAL_SET[0].doc_id]
    assert decide(merged, DEFAULT_POLICY).route == "redact"


async def test_merge_keeps_the_higher_privileged_probability() -> None:
    primary = StubClassifier(band_result(responsive=1.0, privileged=0.15))
    second = StubClassifier(band_result(privileged=0.05))
    merged = await CascadeClassifier(primary, second, DEFAULT_POLICY).classify(
        EVAL_SET[0], "p"
    )
    privileged = merged.answers["privileged"]
    assert isinstance(privileged, PredicateAnswer)
    assert privileged.probability == 0.15
    assert merged.answers["responsive"].score == 2.0  # type: ignore[union-attr]


async def test_cascade_keeps_the_primary_when_the_second_opinion_fails() -> None:
    failing = StubClassifier(ClassifierResult(engine="llm", model="m", error="down"))
    primary = StubClassifier(band_result(responsive=1.0))
    merged = await CascadeClassifier(primary, failing, DEFAULT_POLICY).classify(
        EVAL_SET[0], "p"
    )
    assert merged.error is None
    assert merged.answers["responsive"].score == 1.0  # type: ignore[union-attr]


async def test_prior_production_gate_short_circuits_duplicates() -> None:
    inner = StubClassifier(band_result())
    gate = PriorProductionGate(inner)
    dup = next(p for p in EVAL_SET if p.doc_id == "ARC-000572")
    result = await gate.classify(dup, "p")
    assert (result.agent_disposition, result.cost_usd, inner.seen) == (
        "withhold",
        0,
        [],
    )
    await gate.classify(EVAL_SET[0], "p")
    assert inner.seen == [EVAL_SET[0].doc_id]


async def test_budgeted_classifier_only_adds_spend_even_past_the_limit() -> None:
    budget = BudgetTracker(limit_usd=0.001)
    pricey = band_result().model_copy(update={"cost_usd": 0.002})
    classifier = BudgetedClassifier(StubClassifier(pricey), budget)
    await classifier.classify(EVAL_SET[0], "p")
    await classifier.classify(EVAL_SET[1], "p")
    assert budget.total_usd == pytest.approx(0.004)


def test_tracker_refuses_to_start_when_the_in_flight_reserve_would_exceed() -> None:
    budget = BudgetTracker(limit_usd=1.0)
    budget.add(0.7)
    budget.check(reserve_usd=0.2)
    with pytest.raises(BudgetExceededError, match="in flight"):
        budget.check(reserve_usd=0.31)


async def test_live_run_refuses_up_front_without_calling_the_api() -> None:
    calls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request.url.path)
        return httpx2.Response(200, json=decisions_body(OK_ANSWERS))

    budget = BudgetTracker(limit_usd=1.0)
    budget.add(1.0 - 2 * RUN_PAGE_RESERVE_USD + 0.0001)
    with pytest.raises(BudgetExceededError, match="in flight"):
        await run_live_variant(
            "03a_router_only",
            EVAL_SET,
            client=client_for(handler),
            budget=budget,
            limit=2,
        )
    assert calls == []


async def test_live_run_adds_every_pages_cost() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=decisions_body(OK_ANSWERS))

    budget = BudgetTracker(limit_usd=1.0)
    pages = [p for p in EVAL_SET if p.doc_id not in {"ARC-000572", "ARC-000580"}]
    run = await run_live_variant(
        "03a_router_only", pages, client=client_for(handler), budget=budget, limit=4
    )
    assert len(run.traces) == 4
    assert budget.total_usd == pytest.approx(sum(t.model_cost_usd for t in run.traces))
    assert run.cost_usd == pytest.approx(budget.total_usd)


async def test_live_decisions_run_uses_the_same_trace_schema() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/decisions"):
            return httpx2.Response(200, json=decisions_body(OK_ANSWERS))
        return httpx2.Response(200, json=responses_body({}))

    budget = BudgetTracker()
    budget.add(0.25)
    pages = [p for p in EVAL_SET if p.doc_id not in {"ARC-000572", "ARC-000580"}]
    run = await run_live_variant(
        "04_prompt_caching", pages, client=client_for(handler), budget=budget, limit=3
    )
    assert len(run.traces) == 3
    assert {t.mode for t in run.traces} == {"live"}
    assert run.trace_url is None
    assert budget.total_usd > 0.25
    assert run.cost_usd == pytest.approx(budget.total_usd - 0.25)


async def test_split_workflow_live_run_does_not_add_modelled_background_work() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/decisions"):
            return httpx2.Response(200, json=decisions_body(OK_ANSWERS))
        return httpx2.Response(200, json=responses_body({}))

    pages = [p for p in EVAL_SET if p.doc_id not in {"ARC-000572", "ARC-000580"}]
    run = await run_live_variant(
        "05_split_workflow",
        pages,
        client=client_for(handler),
        budget=BudgetTracker(),
        limit=2,
    )
    assert {t.mode for t in run.traces} == {"live"}
    assert all(t.background_cost_usd == 0 for t in run.traces)
    assert all(t.model_cost_usd == t.sync_cost_usd for t in run.traces)


async def test_trace_summary_separates_model_cost_from_blended_cost() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=decisions_body(OK_ANSWERS))

    run = await run_live_variant(
        "03a_router_only",
        EVAL_SET,
        client=client_for(handler),
        budget=BudgetTracker(),
        limit=2,
    )
    summary = _trace_summary((run, None))
    assert summary["pages"] == 2
    assert summary["model_cost_usd"] == pytest.approx(
        sum(t.model_cost_usd for t in run.traces)
    )
    assert summary["blended_cost_usd"] == pytest.approx(
        sum(t.cost_usd for t in run.traces)
    )
    assert "cost_usd" not in summary


def _llm_rules(body: dict[str, Any]) -> str:
    return str(body["input"][0]["content"][0]["text"])


async def test_cascade_second_opinion_uses_the_variants_question_wording() -> None:
    rules: dict[str, str] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/decisions"):
            ambiguous = [score_answer(1.0, 0.2, [0.3, 0.4, 0.3]), *OK_ANSWERS[1:]]
            return httpx2.Response(200, json=decisions_body(ambiguous))
        rules["text"] = _llm_rules(json.loads(request.content))
        return httpx2.Response(200, json=responses_body({}))

    for variant, expected in (("03b_gist_wording", False), ("04_prompt_caching", True)):
        classifier, _ = build_live_engines(variant, client_for(handler))
        await classifier.classify(EVAL_SET[0], PROTOCOL)
        assert ("only appears on cc" in rules["text"]) is expected


async def test_budget_error_propagates_out_of_a_live_run() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=decisions_body(OK_ANSWERS))

    budget = BudgetTracker(limit_usd=0.0)
    with pytest.raises(BudgetExceededError):
        await run_live_variant(
            "03a_router_only",
            EVAL_SET,
            client=client_for(handler),
            budget=budget,
            limit=2,
        )


def test_batch_rows_and_jsonl(tmp_path: Path) -> None:
    rows = build_background_batch(EVAL_SET[:2])
    assert len(rows) == 6
    row = rows[0]
    assert (row["method"], row["url"], row["body"]["model"]) == (
        "POST",
        "/v1/responses",
        "gpt-6-luna",
    )
    assert row["body"]["reasoning"] == {"effort": "none"}
    assert row["custom_id"] == f"{EVAL_SET[0].doc_id}:privilege_log"
    path = write_batch_jsonl(rows, tmp_path / "out" / "batch.jsonl")
    lines = path.read_text().splitlines()
    assert len(lines) == 6
    assert json.loads(lines[1])["custom_id"].endswith("qa_sample")


def test_privilege_log_only_for_escalated_pages_when_traces_are_given() -> None:
    page = EVAL_SET[0]
    result = _result(page, escalated=False)
    trace = trace_from_result(page, result, RunInfo("v", "V", "simulated"))
    rows = build_background_batch([page], [trace])
    assert [r["custom_id"].split(":")[1] for r in rows] == [
        "qa_sample",
        "production_tags",
    ]
    plain = background_work(page, escalated=False)
    full = background_work(page, escalated=True)
    assert 0 < plain.tokens < full.tokens
    assert plain.cost_usd < full.cost_usd


def _result(page: Any, *, escalated: bool) -> Any:
    from doc_review_opt.models import PageResult

    return PageResult(
        doc_id=page.doc_id,
        decision="produce",
        route="produce",
        produced_text=page.text,
        escalated=escalated,
        classification=ClassifierResult(engine="agent", model="gpt-6-luna"),
    )


def judge_trace() -> Any:
    page = EVAL_SET[0]
    return page, trace_from_result(
        page, _result(page, escalated=False), RunInfo("v", "V", "live")
    )


async def test_judge_grades_with_a_strict_schema_and_no_retries() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json=text_response(json.dumps({"passed": True, "reason": "defensible"})),
        )

    page, trace = judge_trace()
    grade = await judge_page(page, trace, client=client_for(handler))
    body = captured["body"]
    assert body["model"] == JUDGE_MODEL == "gpt-6.1-terra"
    assert body["text"]["format"]["strict"] is True
    assert body["reasoning"] == {"effort": "none"}
    assert "expected_route" not in json.dumps(body)
    assert (grade.judge_status, grade.passed) == ("graded", True)
    assert grade.judge_cost_usd is not None
    assert grade.judge_cost_usd > 0


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"passed": "yes", "reason": "x"}',
        '{"passed": true, "reason": "  "}',
        '{"passed": true}',
    ],
)
async def test_invalid_judge_output_is_an_error_not_a_pass(text: str) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=text_response(text))

    page, trace = judge_trace()
    grade = await judge_page(page, trace, client=client_for(handler))
    assert grade.judge_status == "error"
    assert grade.passed is None
    assert grade.judge_cost_usd is not None


def test_disposition_parsers() -> None:
    withheld = parse_free_text("FINAL DISPOSITION: withhold")
    assert withheld is not None
    assert (withheld.route, withheld.redact) == ("withhold", None)
    redacted = parse_free_text("final disposition: PRODUCE; redaction: true")
    assert redacted is not None
    assert (redacted.route, redacted.redact) == ("redact", True)
    assert parse_free_text("nothing here") is None
    unclear = parse_free_text("FINAL DISPOSITION: produce\n\nRedaction required: yes")
    assert unclear is not None
    assert (unclear.route, unclear.redact) == ("redact", None)
    kept = parse_free_text("FINAL DISPOSITION: produce; redact: no")
    assert kept is not None
    assert (kept.route, kept.redact) == ("produce", False)
    reply = parse_json_disposition(
        '{"disposition": "attorney_review", "redact": false, "reason": "r"}'
    )
    assert reply is not None
    assert (reply.route, reply.redact) == ("attorney_review", False)
    assert parse_json_disposition("{}") is None

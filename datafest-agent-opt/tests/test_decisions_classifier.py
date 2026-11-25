"""Tests for the Decisions and LLM classifiers over a mock transport."""

import json
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from doc_review_opt.agent.classifiers.decisions import (
    GIST_PRIVILEGE_INSTRUCTIONS,
    PROTOCOL_PRIVILEGE_INSTRUCTIONS,
    QUESTIONS,
    DecisionsClassifier,
    build_decisions_request,
    questions,
)
from doc_review_opt.agent.classifiers.llm_probability import (
    LLMProbabilityClassifier,
    confidence,
    normalize,
)
from doc_review_opt.agent.routing import route
from doc_review_opt.fixtures import PROTOCOL
from doc_review_opt.models import PredicateAnswer, RefusalAnswer, ScoreAnswer
from doc_review_opt.pricing import UnknownModelError
from tests.helpers import PAGE

USAGE = {
    "input_tokens": 2000,
    "input_tokens_details": {"cached_tokens": 1500, "cache_write_tokens": 0},
    "output_tokens": 12,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 2012,
}


def score_answer(score: float, confidence: float, probs: list[float]) -> dict[str, Any]:
    return {
        "type": "score",
        "name": "responsive",
        "score": score,
        "confidence": confidence,
        "probabilities": [
            {"value": i, "label": f"l{i}", "probability": p}
            for i, p in enumerate(probs)
        ],
    }


def predicate(name: str, probability: float) -> dict[str, Any]:
    return {"type": "predicate", "name": name, "probability": probability}


def decisions_body(answers: list[dict[str, Any]]) -> dict[str, Any]:
    return {"model": "gpt-6-luna", "answers": answers, "usage": USAGE}


def client_for(handler: Any) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )


OK_ANSWERS = [
    score_answer(1.9, 0.85, [0.02, 0.08, 0.90]),
    predicate("privileged", 0.03),
    predicate("contains_pii", 0.02),
]


async def test_decisions_request_shape_and_parsing() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200, json=decisions_body(OK_ANSWERS), headers={"x-request-id": "req_1"}
        )

    classifier = DecisionsClassifier(client_for(handler), cache_key="matter-24-CV-118")
    result = await classifier.classify(PAGE, PROTOCOL)

    body = captured["body"]
    assert captured["url"].endswith("/decisions")
    assert body["model"] == "gpt-6-luna"
    assert body["prompt_cache_key"] == "matter-24-CV-118"
    questions = {q["name"]: q for q in body["questions"]}
    assert [q["name"] for q in body["questions"]] == [
        "responsive",
        "privileged",
        "contains_pii",
    ]
    assert questions["responsive"]["type"] == "score"
    assert len(questions["responsive"]["levels"]) == 3
    assert questions["privileged"]["type"] == "predicate"
    parts = body["input"][0]["content"]
    assert parts[0]["text"].startswith("<review_protocol>")
    assert PAGE.text in parts[-1]["text"]
    assert "expected_route" not in json.dumps(body)

    assert result.error is None
    assert result.answers["privileged"] == PredicateAnswer(
        name="privileged", probability=0.03
    )
    responsive = result.answers["responsive"]
    assert isinstance(responsive, ScoreAnswer)
    assert responsive.probabilities == (0.02, 0.08, 0.90)
    assert result.usage.cached == 1500
    assert result.cost_usd == pytest.approx((500 * 0.10 + 1500 * 0.01) / 1e6)
    assert route(result) == "produce"


def test_cache_key_is_optional() -> None:
    assert "prompt_cache_key" not in build_decisions_request(PAGE, PROTOCOL)


def privilege_text(wording: str, built: tuple[dict[str, Any], ...]) -> str:
    return str(next(q for q in built if q["name"] == "privileged")["instructions"])


def test_questions_wording_changes_only_the_privileged_instructions() -> None:
    gist, protocol = questions("gist"), questions("protocol")
    assert privilege_text("gist", gist) == GIST_PRIVILEGE_INSTRUCTIONS
    assert privilege_text("protocol", protocol) == PROTOCOL_PRIVILEGE_INSTRUCTIONS
    assert PROTOCOL_PRIVILEGE_INSTRUCTIONS.startswith(GIST_PRIVILEGE_INSTRUCTIONS)
    assert "cc of" in PROTOCOL_PRIVILEGE_INSTRUCTIONS
    assert "cc of" not in GIST_PRIVILEGE_INSTRUCTIONS
    assert [q for q in gist if q["name"] != "privileged"] == [
        q for q in protocol if q["name"] != "privileged"
    ]
    assert protocol == QUESTIONS


def test_review_protocol_states_the_same_cc_rule() -> None:
    assert "not privileged merely because an attorney is copied (cc)" in PROTOCOL


def test_request_carries_protocol_wording_by_default() -> None:
    request = build_decisions_request(PAGE, PROTOCOL)
    assert privilege_text("protocol", tuple(request["questions"])) == (
        PROTOCOL_PRIVILEGE_INSTRUCTIONS
    )
    gist = build_decisions_request(PAGE, PROTOCOL, wording="gist")
    assert privilege_text("gist", tuple(gist["questions"])) == (
        GIST_PRIVILEGE_INSTRUCTIONS
    )


async def test_classifier_sends_the_wording_it_was_given() -> None:
    sent: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(privilege_text("", tuple(body["questions"])))
        return httpx2.Response(200, json=decisions_body(OK_ANSWERS))

    await DecisionsClassifier(client_for(handler)).classify(PAGE, PROTOCOL)
    await DecisionsClassifier(client_for(handler), wording="gist").classify(
        PAGE, PROTOCOL
    )
    assert sent == [PROTOCOL_PRIVILEGE_INSTRUCTIONS, GIST_PRIVILEGE_INSTRUCTIONS]


async def test_refusal_answer_is_typed_and_fails_safe() -> None:
    answers = [OK_ANSWERS[0], {"type": "refusal", "name": "privileged"}, OK_ANSWERS[2]]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=decisions_body(answers))

    result = await DecisionsClassifier(client_for(handler)).classify(PAGE, PROTOCOL)
    assert result.answers["privileged"] == RefusalAnswer(name="privileged")
    assert route(result) == "attorney_review"


async def test_mismatched_answer_names_give_an_error_result_with_request_id() -> None:
    answers = [
        OK_ANSWERS[0],
        predicate("contains_pii", 0.1),
        predicate("privileged", 0),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json=decisions_body(answers), headers={"x-request-id": "req_9"}
        )

    result = await DecisionsClassifier(client_for(handler)).classify(PAGE, PROTOCOL)
    assert result.error is not None
    assert "unmatched_answers" in result.error
    assert "req_9" in result.error
    assert route(result) == "attorney_review"


async def test_http_error_gives_an_error_result() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"error": {"message": "down"}})

    result = await DecisionsClassifier(client_for(handler)).classify(PAGE, PROTOCOL)
    assert result.error is not None
    assert route(result) == "attorney_review"


def responses_body(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6-luna",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(payload),
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 3000,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 60,
            "output_tokens_details": {"reasoning_tokens": 20},
            "total_tokens": 3060,
        },
    }


async def test_llm_classifier_request_and_probability_math() -> None:
    captured: dict[str, Any] = {}
    payload = {
        "responsive_probabilities": [0.1, 0.2, 0.7],
        "privileged_probability": 0.04,
        "contains_pii_probability": 1.4,
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=responses_body(payload))

    classifier = LLMProbabilityClassifier(client_for(handler), "gpt-6-luna")
    result = await classifier.classify(PAGE, PROTOCOL)

    body = captured["body"]
    assert captured["url"].endswith("/responses")
    assert body["model"] == "gpt-6-luna"
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["verbosity"] == "low"
    assert body["reasoning"] == {"effort": "none"}
    assert "max_output_tokens" in body
    responsive = result.answers["responsive"]
    assert isinstance(responsive, ScoreAnswer)
    assert responsive.score == pytest.approx(1.6)
    assert responsive.confidence == pytest.approx((0.7 - 1 / 3) / (2 / 3))
    assert result.answers["contains_pii"] == PredicateAnswer(
        name="contains_pii", probability=1.0
    )
    assert result.usage.reasoning == 20
    assert result.cost_usd == pytest.approx((3000 * 0.10 + 60 * 0.50) / 1e6)


async def test_llm_classifier_rules_follow_the_question_wording() -> None:
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content)["input"][0]["content"][0]["text"])
        return httpx2.Response(200, json=responses_body({}))

    for wording in ("gist", "protocol"):
        classifier = LLMProbabilityClassifier(
            client_for(handler), "gpt-6.1-terra", wording=wording
        )
        await classifier.classify(PAGE, PROTOCOL)
    gist, protocol = seen
    assert "only appears on cc" not in gist
    assert "only appears on cc" in protocol


async def test_choice_answers_are_an_unusable_reply() -> None:
    choice = {
        "type": "choice",
        "name": "privileged",
        "choice": "yes",
        "confidence": 0.9,
        "probabilities": [{"value": "yes", "label": "yes", "probability": 0.9}],
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json=decisions_body([OK_ANSWERS[0], choice, OK_ANSWERS[2]])
        )

    result = await DecisionsClassifier(client_for(handler)).classify(PAGE, PROTOCOL)
    assert result.error is not None
    assert "unexpected_answer" in result.error


async def test_llm_classifier_bad_json_gives_an_error_result() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=responses_body({"unexpected": 1}))

    classifier = LLMProbabilityClassifier(client_for(handler), "gpt-6.1-terra")
    result = await classifier.classify(PAGE, PROTOCOL)
    assert result.error is not None
    assert route(result) == "attorney_review"


def test_llm_classifier_rejects_unpriced_models() -> None:
    with pytest.raises(UnknownModelError):
        LLMProbabilityClassifier(
            client_for(lambda r: httpx2.Response(200)), "gpt-unpriced-model"
        )


async def test_llm_classifier_levels_follow_the_decisions_rubric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def four_levels(wording: Any) -> Any:
        first, *rest = questions(wording)
        extra = {"label": "extra", "description": "Extra level."}
        return ({**first, "levels": [*first["levels"], extra]}, *rest)

    monkeypatch.setattr(
        "doc_review_opt.agent.classifiers.llm_probability.questions", four_levels
    )
    seen: list[str] = []
    reply = {
        "responsive_probabilities": [0.1, 0.2, 0.3, 0.4],
        "privileged_probability": 0.0,
        "contains_pii_probability": 0.0,
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content)["input"][0]["content"][0]["text"])
        return httpx2.Response(200, json=responses_body(reply))

    classifier = LLMProbabilityClassifier(client_for(handler), "gpt-6.1-terra")
    result = await classifier.classify(PAGE, PROTOCOL)
    assert "rubric level 0, 1, 2, 3," in seen[0]
    responsive = result.answers["responsive"]
    assert isinstance(responsive, ScoreAnswer)
    assert responsive.probabilities == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert responsive.score == pytest.approx(2.0)


def test_normalize_pads_clamps_and_rescales() -> None:
    assert normalize([2.0, 2.0], 3) == [0.5, 0.5, 0.0]
    assert normalize([-1.0, 0.0, 0.0], 3) == pytest.approx([1 / 3] * 3)
    assert normalize([0.2, 0.2, 0.2, 0.9], 3) == pytest.approx([1 / 3] * 3)


def test_confidence_is_zero_for_uniform_and_one_for_peaked() -> None:
    assert confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert confidence([0.0, 0.0, 1.0]) == pytest.approx(1.0)

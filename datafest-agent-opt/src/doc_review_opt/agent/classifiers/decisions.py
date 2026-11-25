"""The Decisions API classifier: questions, request builder and reply parser."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Literal

from langsmith import traceable
from openai import AsyncOpenAI, OpenAIError
from openai.types.decision import (
    AnswerAnswerResourcePredicate,
    AnswerAnswerResourceRefusal,
    AnswerAnswerResourceScore,
)
from openai.types.decision import Decision as DecisionReply

from doc_review_opt.config import get_settings
from doc_review_opt.exceptions import UnusableReplyError
from doc_review_opt.models import (
    Answer,
    ClassifierResult,
    PredicateAnswer,
    RefusalAnswer,
    ReviewPage,
    ScoreAnswer,
    TokenUsage,
)
from doc_review_opt.pricing import (
    DECISIONS_PRICE_KEY,
    estimate_cost_usd,
)

logger = logging.getLogger(__name__)


SETTINGS = get_settings()


DECISIONS_MODEL = SETTINGS.models.decisions


GIST_PRIVILEGE_INSTRUCTIONS = (
    "Was this page written by or to a lawyer, or does it convey a lawyer's "
    "legal advice? Answer no for ordinary business, finance, or accounting "
    "discussion, even when the subject is litigation-sensitive.\n\n"
    "True when: a named attorney is the author or recipient, or the text "
    "relays legal advice or litigation strategy.\n"
    "False when: business, finance, operations, or accounting content with "
    "no attorney involved -- including discussion of restatements, audits, "
    "or regulatory exposure."
)


PROTOCOL_PRIVILEGE_INSTRUCTIONS = (
    GIST_PRIVILEGE_INSTRUCTIONS
    + "\nAn attorney who only appears on cc of ordinary business, finance or "
    "operations mail does not make the page privileged unless the text itself "
    "relays legal advice or litigation strategy."
)


Wording = Literal["gist", "protocol"]


ORIGINAL_WORDING: Wording = "gist"


ORIGINAL_PRIVILEGE_INSTRUCTIONS = GIST_PRIVILEGE_INSTRUCTIONS


PRIVILEGE_INSTRUCTIONS: dict[Wording, str] = {
    "gist": GIST_PRIVILEGE_INSTRUCTIONS,
    "protocol": PROTOCOL_PRIVILEGE_INSTRUCTIONS,
}


def questions(wording: Wording) -> tuple[dict[str, Any], ...]:
    privilege_instructions = PRIVILEGE_INSTRUCTIONS[wording]
    return (
        {
            "type": "score",
            "name": "responsive",
            "instructions": (
                "How relevant is this page to the matter described in the review "
                "protocol?"
            ),
            "levels": [
                {
                    "label": "not_responsive",
                    "description": "Not responsive: unrelated to the matter.",
                },
                {
                    "label": "possibly_responsive",
                    "description": (
                        "Possibly responsive: touches on the matter indirectly."
                    ),
                },
                {
                    "label": "responsive",
                    "description": "Responsive: directly concerns the matter.",
                },
            ],
        },
        {
            "type": "predicate",
            "name": "privileged",
            "instructions": privilege_instructions,
        },
        {
            "type": "predicate",
            "name": "contains_pii",
            "instructions": (
                "Does this page contain personal information such as contact details, "
                "account numbers, or health information?"
            ),
        },
    )


QUESTIONS = questions("protocol")


QUESTION_NAMES = tuple(question["name"] for question in QUESTIONS)


def page_block(page: ReviewPage) -> str:
    header = [
        f"doc_id: {page.doc_id}",
        f"custodian: {page.custodian}",
        f"from: {page.author}",
        f"to: {', '.join(page.recipients) or '-'}",
        f"cc: {', '.join(page.cc) or '-'}",
    ]
    return "<page>\n" + "\n".join(header) + f"\n\n{page.text}\n</page>"


def protocol_block(protocol: str) -> str:
    return f"<review_protocol>\n{protocol}\n</review_protocol>"


def build_decisions_request(
    page: ReviewPage,
    protocol: str,
    *,
    cache_key: str | None = None,
    wording: Wording = "protocol",
) -> dict[str, Any]:
    parts = [
        {"type": "input_text", "text": protocol_block(protocol)},
        {"type": "input_text", "text": page_block(page)},
    ]
    request: dict[str, Any] = {
        "model": DECISIONS_MODEL,
        "questions": list(questions(wording)),
        "input": [{"role": "user", "content": parts}],
    }
    if cache_key is not None:
        request["prompt_cache_key"] = cache_key
    return request


def trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    page = inputs.get("page")
    shown = (
        page.model_dump(exclude={"labels"}) if isinstance(page, ReviewPage) else page
    )
    return {"page": shown, "protocol_chars": len(str(inputs.get("protocol", "")))}


def _parse_answer(answer: object, name: str, where: str) -> Answer:
    if isinstance(answer, AnswerAnswerResourcePredicate):
        return PredicateAnswer(name=name, probability=answer.probability)
    if isinstance(answer, AnswerAnswerResourceScore):
        ordered = sorted(answer.probabilities, key=lambda p: p.value)
        return ScoreAnswer(
            name=name,
            score=answer.score,
            confidence=answer.confidence,
            probabilities=tuple(p.probability for p in ordered),
        )
    if isinstance(answer, AnswerAnswerResourceRefusal):
        return RefusalAnswer(name=name)
    raise UnusableReplyError(f"unexpected_answer name={name} {where}")


def _decision_usage(reply: DecisionReply) -> TokenUsage:
    usage = reply.usage
    return TokenUsage(
        input=usage.input_tokens,
        cached=usage.input_tokens_details.cached_tokens,
        cache_write=usage.input_tokens_details.cache_write_tokens,
        output=usage.output_tokens,
        reasoning=usage.output_tokens_details.reasoning_tokens,
    )


def parse_decision(reply: object, names: list[str], where: str) -> dict[str, Answer]:
    if not isinstance(reply, DecisionReply) or not isinstance(reply.answers, list):
        raise UnusableReplyError(f"invalid_body {where}")
    if [getattr(answer, "name", None) for answer in reply.answers] != names:
        raise UnusableReplyError(f"unmatched_answers {where}")
    return {
        name: _parse_answer(answer, name, where)
        for name, answer in zip(names, reply.answers, strict=True)
    }


class DecisionsClassifier:
    engine = "decisions"

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        cache_key: str | None = None,
        wording: Wording = "protocol",
    ) -> None:
        self._client = client
        self._cache_key = cache_key
        self._wording: Wording = wording
        self._traced = traceable(
            run_type="llm",
            name="decisions_classify",
            metadata={"engine": self.engine, "model": DECISIONS_MODEL},
            process_inputs=trace_inputs,
        )(self._classify)

    async def classify(self, page: ReviewPage, protocol: str) -> ClassifierResult:
        result: ClassifierResult = await self._traced(page=page, protocol=protocol)
        return result

    async def _classify(self, page: ReviewPage, protocol: str) -> ClassifierResult:
        request = build_decisions_request(
            page, protocol, cache_key=self._cache_key, wording=self._wording
        )
        started = perf_counter()
        usage = TokenUsage()
        try:
            raw = await self._client.decisions.with_raw_response.create(**request)
            where = f"model={DECISIONS_MODEL} request_id={raw.request_id}"
            reply = raw.parse()
            usage = _decision_usage(reply)
            answers = parse_decision(reply, list(QUESTION_NAMES), where)
        except (OpenAIError, UnusableReplyError, ValueError) as error:
            logger.warning("decisions call failed doc_id=%s: %s", page.doc_id, error)
            return self._failed(error, perf_counter() - started, usage)
        return ClassifierResult(
            engine=self.engine,
            model=DECISIONS_MODEL,
            answers=answers,
            usage=usage,
            latency_s=perf_counter() - started,
            cost_usd=estimate_cost_usd(DECISIONS_PRICE_KEY, usage),
        )

    def _failed(
        self, error: Exception, latency_s: float, usage: TokenUsage
    ) -> ClassifierResult:
        request_id = getattr(error, "request_id", None)
        suffix = f" request_id={request_id}" if request_id else ""
        return ClassifierResult(
            engine=self.engine,
            model=DECISIONS_MODEL,
            usage=usage,
            latency_s=latency_s,
            cost_usd=estimate_cost_usd(DECISIONS_PRICE_KEY, usage),
            error=f"{type(error).__name__}: {error}{suffix}",
        )


__all__ = [
    "DECISIONS_MODEL",
    "GIST_PRIVILEGE_INSTRUCTIONS",
    "ORIGINAL_PRIVILEGE_INSTRUCTIONS",
    "ORIGINAL_WORDING",
    "PROTOCOL_PRIVILEGE_INSTRUCTIONS",
    "QUESTIONS",
    "DecisionsClassifier",
    "build_decisions_request",
]

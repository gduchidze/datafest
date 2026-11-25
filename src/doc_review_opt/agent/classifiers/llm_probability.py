"""An LLM classifier that reports its own probabilities, the Decisions API's rival."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from langsmith import traceable
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from doc_review_opt.agent.classifiers.decisions import (
    Wording,
    page_block,
    protocol_block,
    questions,
    trace_inputs,
)
from doc_review_opt.agent.classifiers.responses import (
    completed_text,
    usage_from_response,
)
from doc_review_opt.config import get_settings
from doc_review_opt.exceptions import UnusableReplyError
from doc_review_opt.models import (
    Answer,
    ClassifierResult,
    PredicateAnswer,
    ReviewPage,
    ScoreAnswer,
    TokenUsage,
)
from doc_review_opt.pricing import (
    MODEL_PRICES_USD_PER_1M,
    UnknownModelError,
    estimate_cost_usd,
)

logger = logging.getLogger(__name__)

SETTINGS = get_settings()
MIN_LEVELS = 2


class _LLMProbabilities(BaseModel):
    responsive_probabilities: list[float]
    privileged_probability: float
    contains_pii_probability: float


LLM_PROBABILITIES_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "page_probabilities",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "responsive_probabilities": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Probability of each rubric level, in order; sums to 1.",
            },
            "privileged_probability": {"type": "number"},
            "contains_pii_probability": {"type": "number"},
        },
        "required": [
            "responsive_probabilities",
            "privileged_probability",
            "contains_pii_probability",
        ],
        "additionalProperties": False,
    },
}


def _describe_questions(wording: Wording) -> str:
    lines: list[str] = []
    for question in questions(wording):
        if question["type"] == "score":
            levels = "; ".join(
                f"{i}={level['description']}"
                for i, level in enumerate(question["levels"])
            )
            lines.append(
                f"- {question['name']} (score): {question['instructions']} {levels}"
            )
        else:
            lines.append(f"- {question['name']} (yes/no): {question['instructions']}")
    return "\n".join(lines)


def responsive_levels(wording: Wording) -> int:
    return len(questions(wording)[0]["levels"])


def classifier_rules(wording: Wording) -> str:
    level_ids = ", ".join(str(i) for i in range(responsive_levels(wording)))
    return (
        "You are a document classifier for e-discovery review. Answer the questions "
        "about the page using the review protocol. Report genuine uncertainty in the "
        "probabilities; do not round to 0 or 1.\n"
        "- responsive_probabilities: one probability for each rubric level "
        f"{level_ids}, in order, summing to 1.\n"
        "- privileged_probability: chance the answer to `privileged` is yes.\n"
        "- contains_pii_probability: chance the answer to `contains_pii` is yes.\n\n"
        "QUESTIONS:\n" + _describe_questions(wording)
    )


def normalize(raw: list[float], levels: int) -> list[float]:
    values = [max(0.0, float(v)) for v in raw][:levels]
    values += [0.0] * (levels - len(values))
    total = sum(values)
    return [v / total for v in values] if total > 0 else [1.0 / levels] * levels


def confidence(probabilities: list[float]) -> float:
    n = len(probabilities)
    if n < MIN_LEVELS:
        return 1.0
    chance = 1.0 / n
    return max(0.0, min(1.0, (max(probabilities) - chance) / (1.0 - chance)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class LLMProbabilityClassifier:
    engine = "llm"

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        wording: Wording = "protocol",
    ) -> None:
        if model not in MODEL_PRICES_USD_PER_1M:
            raise UnknownModelError(f"no price for model {model!r}")
        self._client = client
        self._model = model
        self._rules = classifier_rules(wording)
        self._levels = responsive_levels(wording)
        self._traced = traceable(
            run_type="llm",
            name="llm_probability_classify",
            metadata={"engine": self.engine, "model": model},
            process_inputs=trace_inputs,
        )(self._classify)

    async def classify(self, page: ReviewPage, protocol: str) -> ClassifierResult:
        result: ClassifierResult = await self._traced(page=page, protocol=protocol)
        return result

    def _request(self, page: ReviewPage, protocol: str) -> dict[str, Any]:
        def text(value: str) -> list[dict[str, str]]:
            return [{"type": "input_text", "text": value}]

        return {
            "model": self._model,
            "input": [
                {
                    "role": "developer",
                    "content": text(f"{self._rules}\n\n{protocol_block(protocol)}"),
                },
                {"role": "user", "content": text(page_block(page))},
            ],
            "text": {"format": LLM_PROBABILITIES_FORMAT, "verbosity": "low"},
            "reasoning": {"effort": SETTINGS.live.reasoning_effort},
            "max_output_tokens": SETTINGS.live.llm_max_output_tokens,
        }

    async def _classify(self, page: ReviewPage, protocol: str) -> ClassifierResult:
        started = perf_counter()
        usage = TokenUsage()
        try:
            response = await self._client.responses.create(
                **self._request(page, protocol)
            )
            usage = usage_from_response(response)
            reply = _LLMProbabilities.model_validate_json(completed_text(response))
        except (OpenAIError, UnusableReplyError, ValidationError) as error:
            logger.warning("llm call failed doc_id=%s: %s", page.doc_id, error)
            return ClassifierResult(
                engine=self.engine,
                model=self._model,
                usage=usage,
                latency_s=perf_counter() - started,
                cost_usd=estimate_cost_usd(self._model, usage),
                error=f"{type(error).__name__}: {error}",
            )
        return ClassifierResult(
            engine=self.engine,
            model=self._model,
            answers=self._answers(reply),
            usage=usage,
            latency_s=perf_counter() - started,
            cost_usd=estimate_cost_usd(self._model, usage),
        )

    def _answers(self, reply: _LLMProbabilities) -> dict[str, Answer]:
        probabilities = normalize(reply.responsive_probabilities, self._levels)
        return {
            "responsive": ScoreAnswer(
                name="responsive",
                score=sum(i * p for i, p in enumerate(probabilities)),
                confidence=confidence(probabilities),
                probabilities=tuple(probabilities),
            ),
            "privileged": PredicateAnswer(
                name="privileged",
                probability=_clamp01(reply.privileged_probability),
            ),
            "contains_pii": PredicateAnswer(
                name="contains_pii",
                probability=_clamp01(reply.contains_pii_probability),
            ),
        }


__all__ = ["LLMProbabilityClassifier", "classifier_rules"]

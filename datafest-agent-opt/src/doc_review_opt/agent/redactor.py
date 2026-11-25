"""The OpenAI redactor that replaces personal information with a token."""

from __future__ import annotations

from time import perf_counter

from langsmith import traceable
from openai import AsyncOpenAI, OpenAIError

from doc_review_opt.agent.classifiers.responses import (
    completed_text,
    usage_from_response,
)
from doc_review_opt.agent.prompts import (
    REDACTION_PROMPT,
)
from doc_review_opt.config import get_settings
from doc_review_opt.exceptions import BilledCallError, UnusableReplyError
from doc_review_opt.models import (
    RedactionResult,
    ReviewPage,
)
from doc_review_opt.pricing import estimate_cost_usd

SETTINGS = get_settings()


REDACTION_MODEL = SETTINGS.models.redaction


class RedactionError(BilledCallError): ...


class OpenAIRedactor:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client
        self._traced = traceable(
            run_type="llm",
            name="redact_page",
            metadata={"engine": "redactor", "model": REDACTION_MODEL},
        )(self._redact)

    async def redact(self, page: ReviewPage) -> RedactionResult:
        result: RedactionResult = await self._traced(page.text)
        return result

    async def _redact(self, text: str) -> RedactionResult:
        started = perf_counter()
        try:
            response = await self._client.responses.create(
                model=REDACTION_MODEL,
                input=REDACTION_PROMPT + text,
                reasoning={"effort": SETTINGS.live.reasoning_effort},
                text={"verbosity": "low"},
                max_output_tokens=SETTINGS.live.redaction_max_output_tokens,
            )
        except OpenAIError as error:
            raise RedactionError(str(error)) from error
        usage = usage_from_response(response)
        try:
            redacted = completed_text(response)
        except UnusableReplyError as error:
            cost = estimate_cost_usd(REDACTION_MODEL, usage)
            raise RedactionError(str(error), usage, cost) from error
        return RedactionResult(
            text=redacted,
            model=REDACTION_MODEL,
            usage=usage,
            latency_s=perf_counter() - started,
            cost_usd=estimate_cost_usd(REDACTION_MODEL, usage),
        )

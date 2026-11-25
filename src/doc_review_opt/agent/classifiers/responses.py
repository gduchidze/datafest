"""Helpers shared by the callers of the Responses API."""

from __future__ import annotations

from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseOutputRefusal,
)

from doc_review_opt.exceptions import UnusableReplyError
from doc_review_opt.models import TokenUsage


def usage_from_response(response: Response) -> TokenUsage:
    usage = response.usage
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input=usage.input_tokens,
        cached=usage.input_tokens_details.cached_tokens,
        cache_write=usage.input_tokens_details.cache_write_tokens,
        output=usage.output_tokens,
        reasoning=usage.output_tokens_details.reasoning_tokens,
    )


def completed_text(response: Response) -> str:
    where = f"response_id={response.id}"
    for item in response.output:
        if isinstance(item, ResponseOutputMessage) and any(
            isinstance(part, ResponseOutputRefusal) for part in item.content
        ):
            raise UnusableReplyError(f"refusal {where}")
    if response.status != "completed":
        reason = (
            response.incomplete_details.reason if response.incomplete_details else None
        )
        raise UnusableReplyError(f"status={response.status} reason={reason} {where}")
    text = response.output_text
    if not text:
        raise UnusableReplyError(f"no_output {where}")
    return text

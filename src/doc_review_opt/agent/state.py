"""Graph state types and the attorney-review interrupt payload."""

from __future__ import annotations

import operator
from typing import (
    Annotated,
    Any,
    NotRequired,
    TypedDict,
)

from doc_review_opt.models import (
    BilledSpend,
    ClassifierResult,
    PageResult,
    PredicateAnswer,
    ReviewPage,
    Route,
    ScoreAnswer,
)

InterruptPayload = dict[str, Any]


class PageState(TypedDict):
    doc_id: str
    page: ReviewPage
    protocol: str
    classification: ClassifierResult
    route: Route
    reason: str
    failed_redaction: NotRequired[BilledSpend]
    results: Annotated[list[PageResult], operator.add]


class BatchState(TypedDict):
    pages: list[ReviewPage]
    results: Annotated[list[PageResult], operator.add]


def interrupt_payload(
    doc_id: str, reason: str, classification: ClassifierResult
) -> InterruptPayload:
    privileged = classification.answers.get("privileged")
    responsive = classification.answers.get("responsive")
    return {
        "doc_id": doc_id,
        "reason": reason,
        "privileged": (
            privileged.probability if isinstance(privileged, PredicateAnswer) else None
        ),
        "responsive_score": (
            responsive.score if isinstance(responsive, ScoreAnswer) else None
        ),
        "responsive_confidence": (
            responsive.confidence if isinstance(responsive, ScoreAnswer) else None
        ),
    }

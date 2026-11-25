"""Structural types for the classifier and redactor the graph depends on."""

from __future__ import annotations

from typing import (
    Protocol,
)

from doc_review_opt.models import (
    ClassifierResult,
    RedactionResult,
    ReviewPage,
)


class Classifier(Protocol):
    async def classify(
        self, page: ReviewPage, protocol: str, /
    ) -> ClassifierResult: ...


class Redactor(Protocol):
    async def redact(self, page: ReviewPage) -> RedactionResult: ...

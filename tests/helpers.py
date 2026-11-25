"""Shared test doubles: canned classifier results, stub engines and a fake redactor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.models import (
    ClassifierResult,
    PredicateAnswer,
    RedactionResult,
    RefusalAnswer,
    ReviewPage,
    ScoreAnswer,
)

PAGE = EVAL_SET[0]
REDACTED_TEXT = "Contact [REDACTED] at [REDACTED]."


def make_result(
    *, responsive: float, confidence: float, privileged: float, pii: float
) -> ClassifierResult:
    return ClassifierResult(
        engine="stub",
        model="stub-model",
        answers={
            "responsive": ScoreAnswer(
                name="responsive",
                score=responsive,
                confidence=confidence,
                probabilities=(0.1, 0.2, 0.7),
            ),
            "privileged": PredicateAnswer(name="privileged", probability=privileged),
            "contains_pii": PredicateAnswer(name="contains_pii", probability=pii),
        },
    )


def refused(name: str) -> RefusalAnswer:
    return RefusalAnswer(name=name)


class StubClassifier:
    def __init__(
        self,
        result: ClassifierResult | Callable[[ReviewPage], ClassifierResult],
    ) -> None:
        self._result = result
        self.seen: list[str] = []

    async def classify(self, page: ReviewPage, protocol: str) -> ClassifierResult:
        self.seen.append(page.doc_id)
        if callable(self._result):
            return self._result(page)
        return self._result


class FakeRedactor:
    async def redact(self, page: ReviewPage) -> RedactionResult:
        return RedactionResult(text=REDACTED_TEXT, model="fake")


def at(frame: pd.DataFrame, row: Any, column: str) -> Any:
    return frame.loc[row, column]


def row(frame: pd.DataFrame, label: Any) -> Any:
    return frame.loc[label]


def assert_clean_chart(fig: Figure, axes: int) -> None:
    assert isinstance(fig, Figure)
    assert len(fig.axes) == axes
    assert fig.get_suptitle()
    assert all(len(a.get_shared_y_axes().get_siblings(a)) == 1 for a in fig.axes)
    assert len({a.get_position().bounds for a in fig.axes}) == axes
    assert all(not a.spines["top"].get_visible() for a in fig.axes)


def bar_colors(axis: Axes) -> list[str]:
    return [to_hex(patch.get_facecolor()) for patch in axis.patches]

"""Cell constructors shared by the notebook section modules."""

from __future__ import annotations

from typing import Literal, NamedTuple


class Cell(NamedTuple):
    kind: Literal["markdown", "code"]
    source: str
    hidden: bool = False


def md(text: str) -> Cell:
    return Cell("markdown", text.strip("\n"))


def code(text: str, *, hidden: bool = False) -> Cell:
    return Cell("code", text.strip("\n"), hidden)

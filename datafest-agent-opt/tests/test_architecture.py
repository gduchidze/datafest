"""The agent package stays independent of experiment, reporting and plotting code."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "doc_review_opt"
AGENT_FORBIDDEN = (
    "doc_review_opt.experiment",
    "doc_review_opt.reporting",
    "matplotlib",
    "pandas",
    "langsmith.evaluation",
)
FOUNDATION = ("models", "config", "exceptions", "pricing", "fixtures")
FOUNDATION_FORBIDDEN = (
    "doc_review_opt.agent",
    "doc_review_opt.experiment",
    "doc_review_opt.reporting",
)


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def violations(path: Path, forbidden: tuple[str, ...]) -> list[str]:
    return sorted(
        module
        for module in imported_modules(path)
        for banned in forbidden
        if module == banned or module.startswith(f"{banned}.")
    )


def test_agent_modules_exist() -> None:
    assert list((PACKAGE / "agent").rglob("*.py"))


@pytest.mark.parametrize(
    "path",
    sorted((PACKAGE / "agent").rglob("*.py")),
    ids=lambda path: path.relative_to(PACKAGE).as_posix(),
)
def test_agent_does_not_import_the_layers_around_it(path: Path) -> None:
    assert violations(path, AGENT_FORBIDDEN) == []


@pytest.mark.parametrize("name", FOUNDATION)
def test_foundation_does_not_import_upward(name: str) -> None:
    assert violations(PACKAGE / f"{name}.py", FOUNDATION_FORBIDDEN) == []

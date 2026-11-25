"""Read the notebooks distributed with the project for offline tests."""

from pathlib import Path

import nbformat
from nbformat import NotebookNode

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def read_notebook(filename: str) -> NotebookNode:
    notebook: NotebookNode = nbformat.read(NOTEBOOKS / filename, 4)  # type: ignore[no-untyped-call]
    return notebook


def read_demo() -> NotebookNode:
    return read_notebook("optimizing_document_review_agents_demo.ipynb")


def read_full() -> NotebookNode:
    return read_notebook("optimizing_document_review_agents_for_cost_and_quality.ipynb")

"""Build the optimizing-document-review notebook from the section modules."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from scripts import nb_demo, nb_intro, nb_rounds, nb_tradeoffs, nb_wrapup
from scripts.cells import Cell

NOTEBOOK_NAME = "optimizing_document_review_agents_for_cost_and_quality.ipynb"
NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / NOTEBOOK_NAME
DEMO_NOTEBOOK_NAME = "optimizing_document_review_agents_demo.ipynb"
DEMO_NOTEBOOK_PATH = NOTEBOOK_PATH.parent / DEMO_NOTEBOOK_NAME
KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}


def _to_cell(index: int, cell: Cell) -> NotebookNode:
    make = new_markdown_cell if cell.kind == "markdown" else new_code_cell
    node: NotebookNode = make(cell.source, id=f"cell-{index:03d}")  # type: ignore[no-untyped-call]
    if cell.hidden:
        node.metadata["jupyter"] = {"source_hidden": True}
    return node


def _assemble(flat: list[Cell]) -> NotebookNode:
    notebook: NotebookNode = new_notebook(  # type: ignore[no-untyped-call]
        cells=[_to_cell(i, cell) for i, cell in enumerate(flat)]
    )
    notebook.metadata["kernelspec"] = KERNELSPEC
    notebook.metadata["language_info"] = {"name": "python"}
    return notebook


def build() -> NotebookNode:
    sections = (
        nb_intro.cells(),
        nb_rounds.cells(),
        nb_tradeoffs.cells(),
        nb_wrapup.cells(),
    )
    return _assemble([cell for section in sections for cell in section])


def build_demo() -> NotebookNode:
    return _assemble(nb_demo.cells())


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    for notebook, path in (
        (build(), NOTEBOOK_PATH),
        (build_demo(), DEMO_NOTEBOOK_PATH),
    ):
        nbformat.write(notebook, path)  # type: ignore[no-untyped-call]
        sys.stdout.write(f"{path}\n")


if __name__ == "__main__":
    main()

"""Executes the generated notebook offline and checks its outputs are clean."""

from __future__ import annotations

from pathlib import Path

import pytest
from nbclient import NotebookClient

from doc_review_opt.reporting.diagrams import list_diagrams
from scripts.build_notebook import build

FLAGS = ("RUN_LIVE_API_CALLS", "RUN_LLM_JUDGE", "LANGSMITH_TRACING")
SECRETS = ("OPENAI_API_KEY", "LANGSMITH_API_KEY")
MIN_CELLS = 100
BAD_MARKERS = ("Deserializing unregistered type", "Warning")
DEMO_ONLY_DIAGRAMS = {"eval_pipeline"}


def test_builder_is_deterministic() -> None:
    first, second = build(), build()
    assert first.cells == second.cells
    assert len(first.cells) >= MIN_CELLS
    assert first.metadata["kernelspec"]["name"] == "python3"


def test_diagrams_are_drawn_cells_not_mermaid_blocks() -> None:
    cells = build().cells
    assert not [c for c in cells if "```mermaid" in c.source]
    drawn = [
        c.source for c in cells if c.cell_type == "code" and "draw_diagram(" in c.source
    ]
    long_diagrams = [n for n in list_diagrams() if n not in DEMO_ONLY_DIAGRAMS]
    assert len(drawn) == len(long_diagrams)
    for name in long_diagrams:
        assert any(f'draw_diagram("{name}")' in source for source in drawn)


def test_notebook_runs_offline_without_errors_or_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for flag in FLAGS:
        monkeypatch.setenv(flag, "false")
    for secret in SECRETS:
        monkeypatch.delenv(secret, raising=False)
    monkeypatch.setenv("DOC_REVIEW_OUTPUT_DIR", str(tmp_path / "outputs"))
    notebook = NotebookClient(
        build(),
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()

    outputs = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    ]
    assert [o for o in outputs if o.output_type == "error"] == []
    streams = [o for o in outputs if o.output_type == "stream"]
    assert [o for o in streams if o.name == "stderr"] == []
    assert not [o for o in streams if any(m in o.text for m in BAD_MARKERS)]
    text = "\n".join(o.text for o in streams)
    assert "Round 1: controls: ACCEPTED" in text
    assert "Checked: no interrupt payload contains page text." in text
    assert (tmp_path / "outputs" / "background_batch.jsonl").exists()

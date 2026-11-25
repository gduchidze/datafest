"""Executes the condensed demo notebook offline and checks its shape and outputs."""

from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib.text import Text
from nbclient import NotebookClient

from doc_review_opt.agent.variants import CONTROLS, SPLIT, VARIANT_LABELS, VARIANT_ORDER
from doc_review_opt.experiment.tradeoffs import plot_engine_tradeoff
from doc_review_opt.reporting.reporting import (
    plot_progress,
    plot_quality_gate,
    plot_round_chart,
    plot_sim_vs_live,
    recorded_live_note,
    recorded_live_summary,
)
from doc_review_opt.reporting.walkthrough import (
    follow_page,
    gate_table,
    holdout_table,
    metrics_table,
)
from tests.notebook_files import read_demo
from tests.simulation_facts import facts

FLAGS = ("RUN_LIVE_API_CALLS", "RUN_LLM_JUDGE", "LANGSMITH_TRACING")
SECRETS = ("OPENAI_API_KEY", "LANGSMITH_API_KEY")
BAD_MARKERS = ("Deserializing unregistered type", "Warning")
MIN_CELLS, MAX_CELLS = 40, 62
MAX_MARKDOWN_CHARS = 700
MAX_CODE_LINES, MAX_SETUP_LINES = 12, 25
DIAGRAMS = (
    "bad_baseline",
    "round1",
    "round2",
    "round3",
    "round4",
    "round5",
    "human_in_the_loop",
    "decisions_or_llm",
    "eval_pipeline",
)
MAX_VISIBLE_CODE_CELLS = 3
MIN_FONT, MIN_TITLE_FONT, MIN_FIGURE_WIDTH, MAX_TABLE_COLUMNS = 12, 14, 12, 8
INTRO_RUN = "3 pages finished, 1 waiting for an attorney: ARC-000284"


def test_demo_size_and_kernel() -> None:
    notebook = read_demo()
    assert MIN_CELLS <= len(notebook.cells) <= MAX_CELLS
    assert notebook.metadata["kernelspec"]["name"] == "python3"


def test_cells_stay_short() -> None:
    cells = read_demo().cells
    markdown = [c for c in cells if c.cell_type == "markdown"]
    code = [c for c in cells if c.cell_type == "code"]
    assert [c.source for c in markdown if len(c.source) > MAX_MARKDOWN_CHARS] == []
    setup, rest = code[0], code[1:]
    assert len(setup.source.splitlines()) <= MAX_SETUP_LINES
    assert [c.source for c in rest if len(c.source.splitlines()) > MAX_CODE_LINES] == []


def test_every_diagram_is_drawn() -> None:
    sources = [c.source for c in read_demo().cells if c.cell_type == "code"]
    for name in DIAGRAMS:
        assert any(f'draw_diagram("{name}")' in source for source in sources)
    assert any("draw_graph(" in source for source in sources)


def test_demo_runs_offline_without_errors_or_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for flag in FLAGS:
        monkeypatch.setenv(flag, "false")
    for secret in SECRETS:
        monkeypatch.delenv(secret, raising=False)
    notebook = NotebookClient(
        read_demo(),
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
    assert not [o for o in streams if "Traceback" in o.text]
    text = "\n".join(o.text for o in streams)
    assert f"{VARIANT_LABELS[CONTROLS]}: ACCEPTED" in text
    for expected in (
        INTRO_RUN,
        "REJECTED",
        "Leak-free on the unseen half: True",
        "def decide(",
        "Offline.",
    ):
        assert expected in text


def test_plumbing_cells_hide_their_source_and_teaching_cells_show_it() -> None:
    cells = [c for c in read_demo().cells if c.cell_type == "code"]

    def hidden(cell: object) -> bool:
        return bool(cell.metadata.get("jupyter", {}).get("source_hidden"))  # type: ignore[attr-defined]

    assert hidden(cells[0])
    shown = [c.source for c in cells if not hidden(c)]
    assert any("inspect.getsource(decide)" in source for source in shown)
    assert any("follow_page(traces" in source for source in shown)
    assert len(shown) <= MAX_VISIBLE_CODE_CELLS
    assert all("draw_diagram" not in source for source in shown)


def test_demo_figures_are_projector_sized() -> None:
    fx = facts()
    summary = fx.summary
    figures = [
        plot_round_chart(summary),
        plot_progress(summary, upto_variant=SPLIT),
        plot_quality_gate(gate_table(list(fx.traces), VARIANT_ORDER), fx.floor),
        plot_engine_tradeoff(fx.engines),
        plot_sim_vs_live(summary, recorded_live_summary(), recorded_live_note()),
    ]
    for fig in figures:
        fig.canvas.draw()
        texts = [t for t in fig.findobj(Text) if t.get_text().strip()]
        assert min(float(t.get_fontsize()) for t in texts) >= MIN_FONT
        title = next(t for t in texts if t.get_text() == fig.get_suptitle())
        assert float(title.get_fontsize()) >= MIN_TITLE_FONT
        assert fig.get_size_inches()[0] >= MIN_FIGURE_WIDTH


def test_demo_tables_stay_narrow() -> None:
    fx = facts()
    traces = list(fx.traces)
    assert len(metrics_table(traces, VARIANT_ORDER).columns) <= MAX_TABLE_COLUMNS
    assert len(follow_page(traces, "ARC-000284").columns) <= MAX_TABLE_COLUMNS
    assert len(holdout_table(fx.holdout).columns) <= MAX_TABLE_COLUMNS

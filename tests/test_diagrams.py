"""Tests for the matplotlib flow diagrams drawn from diagrams.toml."""

from __future__ import annotations

import copy
import tomllib
from importlib.resources import files
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import pytest
from langchain_core.runnables.graph import Graph
from langgraph.graph import END, START, StateGraph
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

from doc_review_opt.agent.graph import build_review_graph, review_checkpointer
from doc_review_opt.config import (
    DiagramSettings,
    get_diagrams,
    get_settings,
    parse_diagrams,
)
from doc_review_opt.exceptions import ConfigError, DiagramLayoutError
from doc_review_opt.reporting import diagrams
from doc_review_opt.reporting.diagrams import (
    DiagramFigure,
    draw_diagram,
    draw_graph,
    graph_spec,
    list_diagrams,
)
from tests.helpers import FakeRedactor, StubClassifier, make_result

EXPECTED = {
    "optimization_loop",
    "notebook_modes",
    "bad_baseline",
    "page_review",
    "round1",
    "round2",
    "round3",
    "round4",
    "round5",
    "human_in_the_loop",
    "decisions_or_llm",
    "eval_pipeline",
}
MAX_BOXES = 10
MIN_FONT = 11
PNG_MAGIC = b"\x89PNG"
NAMES = sorted(EXPECTED)


def raw() -> dict[str, Any]:
    text = files("doc_review_opt").joinpath("diagrams.toml").read_text("utf-8")
    return tomllib.loads(text)


def renderer_of(fig: Figure) -> RendererBase:
    assert isinstance(fig.canvas, FigureCanvasAgg)
    return fig.canvas.get_renderer()  # type: ignore[no-untyped-call, no-any-return]


def test_list_matches_toml() -> None:
    assert list_diagrams() == list(raw()["diagrams"])
    assert set(list_diagrams()) == EXPECTED


@pytest.mark.parametrize("name", NAMES)
def test_diagram_renders_a_figure_without_axes(name: str) -> None:
    fig = draw_diagram(name)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    assert not fig.axes[0].axison
    assert len(get_diagrams().diagrams[name].nodes) <= MAX_BOXES


@pytest.mark.parametrize("name", NAMES)
def test_edges_and_groups_reference_existing_nodes(name: str) -> None:
    spec = get_diagrams().diagrams[name]
    ids = {node.id for node in spec.nodes}
    assert len(ids) == len(spec.nodes)
    assert {e.source for e in spec.edges} | {e.target for e in spec.edges} <= ids
    assert {n for group in spec.groups for n in group.nodes} <= ids


@pytest.mark.parametrize("name", NAMES)
def test_no_label_overflows_its_box(name: str) -> None:
    fig = draw_diagram(name)
    renderer = renderer_of(fig)
    node_ids = {node.id for node in get_diagrams().diagrams[name].nodes}
    boxes = {
        p.get_gid(): p.get_window_extent(renderer)
        for p in fig.axes[0].patches
        if p.get_gid() in node_ids
    }
    texts = {
        t.get_gid(): t.get_window_extent(renderer)
        for t in fig.axes[0].texts
        if t.get_gid() in node_ids
    }
    assert set(boxes) == set(texts) == node_ids
    for node_id, text in texts.items():
        box = boxes[node_id]
        assert box.x0 < text.x0
        assert text.x1 < box.x1
        assert box.y0 < text.y0
        assert text.y1 < box.y1


@pytest.mark.parametrize("name", NAMES)
def test_edge_labels_stay_clear_of_boxes(name: str) -> None:
    fig = draw_diagram(name)
    renderer = renderer_of(fig)
    node_ids = {node.id for node in get_diagrams().diagrams[name].nodes}
    boxes = [
        p.get_window_extent(renderer)
        for p in fig.axes[0].patches
        if p.get_gid() in node_ids
    ]
    labels = [
        t for t in fig.axes[0].texts if t.get_gid() and "->" in (t.get_gid() or "")
    ]
    for label in labels:
        patch = label.get_bbox_patch()
        assert patch is not None
        extent = patch.get_window_extent(renderer)
        assert not any(extent.overlaps(box) for box in boxes)


def test_fonts_are_projector_sized() -> None:
    layout = get_diagrams().layout
    assert layout.font_size >= MIN_FONT
    assert layout.label_font_size >= MIN_FONT
    for text in draw_diagram("page_review").axes[0].texts:
        assert float(text.get_fontsize()) >= MIN_FONT


@pytest.mark.parametrize("name", NAMES)
def test_node_fills_follow_their_kind(name: str) -> None:
    palette = get_settings().palette.model_dump()
    kinds = get_diagrams().kinds
    spec = get_diagrams().diagrams[name]
    patches = {p.get_gid(): p for p in draw_diagram(name).axes[0].patches}
    for node in spec.nodes:
        fill = to_hex(patches[node.id].get_facecolor())
        assert fill == palette[kinds[node.kind].fill]


def test_every_kind_uses_a_palette_color() -> None:
    palette = set(get_settings().palette.model_dump())
    for style in get_diagrams().kinds.values():
        assert {style.fill, style.text} <= palette


MODEL_COLORS = {
    "decisions": "blue",
    "luna": "orange",
    "terra": "aqua",
    "agent": "yellow",
}


def test_model_kinds_are_neutral_with_a_colored_border() -> None:
    palette = get_settings().palette.model_dump()
    kinds = get_diagrams().kinds
    for kind, color in MODEL_COLORS.items():
        assert kinds[kind].fill == "neutral"
        assert kinds[kind].edge == color
    assert {k for k, s in kinds.items() if s.edge} == set(MODEL_COLORS)
    patches = {p.get_gid(): p for p in draw_diagram("round2").axes[0].patches}
    for node in get_diagrams().diagrams["round2"].nodes:
        patch = patches[node.id]
        assert to_hex(patch.get_facecolor()) == palette["neutral"]
        assert to_hex(patch.get_edgecolor()) == palette[MODEL_COLORS[node.kind]]
        assert patch.get_linewidth() == get_diagrams().layout.model_line_width


def test_model_nodes_use_model_kinds() -> None:
    models = get_settings().models
    for spec in get_diagrams().diagrams.values():
        for node in spec.nodes:
            if models.second_opinion in node.label or "Decisions" in node.label:
                assert node.kind in MODEL_COLORS
            if node.kind in MODEL_COLORS:
                assert node.kind != "decisions" or "Decisions" in node.label
    kinds = {n.id: n.kind for n in get_diagrams().diagrams["round2"].nodes}
    assert kinds == {
        "all": "agent",
        "triage": "luna",
        "routine": "luna",
        "risky": "terra",
    }


def test_filled_kinds_are_only_outcomes() -> None:
    filled = {k for k, s in get_diagrams().kinds.items() if s.fill != "neutral"}
    assert filled == {"produce", "withhold", "redact", "attorney"}


def test_dashed_edges_are_drawn_dashed() -> None:
    patches = draw_diagram("round5").axes[0].patches
    arrows = [p for p in patches if isinstance(p, FancyArrowPatch)]
    styles = [p.get_linestyle() for p in arrows]
    assert styles.count("--") == 3
    assert len(arrows) == 5


def test_drawing_leaves_no_pyplot_state() -> None:
    draw_diagram("page_review")
    assert plt.get_fignums() == []


def test_outcome_boxes_use_route_colors() -> None:
    palette = get_settings().palette
    expected = {
        "attorney": palette.yellow,
        "withhold": palette.orange,
        "redact": palette.aqua,
        "produce": palette.blue,
    }
    patches = {p.get_gid(): p for p in draw_diagram("page_review").axes[0].patches}
    for node_id, color in expected.items():
        assert to_hex(patches[node_id].get_facecolor()) == color


def test_policy_placeholders_resolve_from_config() -> None:
    policy = get_settings().policy.calibrated
    labels = " ".join(
        [n.label for n in get_diagrams().diagrams["page_review"].nodes]
        + [e.label for e in get_diagrams().diagrams["page_review"].edges]
    )
    assert "{" not in labels
    assert f"privilege > {policy.privilege_escalation}" in labels
    assert f"PII > {policy.pii_redaction}" in labels
    assert f"{policy.not_responsive} to {policy.responsive}" in labels
    assert get_settings().models.second_opinion in labels
    lanes = " ".join(g.label for g in get_diagrams().diagrams["round5"].groups)
    assert "50% off" in lanes
    floor = get_settings().quality.guardrail_pass_floor
    gate = next(
        n.label
        for n in get_diagrams().diagrams["eval_pipeline"].nodes
        if n.id == "gate"
    )
    assert f"{floor:.0%} floor" in gate
    assert "{" not in gate


def break_diagram(change: Any) -> DiagramSettings:
    data = copy.deepcopy(raw())
    change(data["diagrams"]["round3"])
    return parse_diagrams(data, get_settings())


def test_unknown_placeholder_is_rejected() -> None:
    def change(spec: dict[str, Any]) -> None:
        spec["nodes"][0]["label"] = "{missing}"

    with pytest.raises(ConfigError, match="unknown placeholder"):
        break_diagram(change)


def test_unknown_node_and_kind_are_rejected() -> None:
    def bad_edge(spec: dict[str, Any]) -> None:
        spec["edges"][0]["to"] = "nowhere"

    def bad_kind(spec: dict[str, Any]) -> None:
        spec["nodes"][0]["kind"] = "sparkle"

    with pytest.raises(ConfigError, match="unknown nodes"):
        break_diagram(bad_edge)
    with pytest.raises(ConfigError, match="unknown kinds"):
        break_diagram(bad_kind)


def test_kind_with_unknown_palette_color_is_rejected() -> None:
    data = copy.deepcopy(raw())
    data["kinds"]["step"]["fill"] = "purple"
    with pytest.raises(ConfigError, match="not in"):
        parse_diagrams(data, get_settings())


def test_invalid_shape_is_rejected() -> None:
    data = copy.deepcopy(raw())
    del data["layout"]
    with pytest.raises(ConfigError, match=r"invalid diagrams\.toml"):
        parse_diagrams(data, get_settings())


def test_overlapping_nodes_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def squash(spec: dict[str, Any]) -> None:
        spec["nodes"][1]["col"] = spec["nodes"][0]["col"]
        spec["nodes"][1]["row"] = spec["nodes"][0]["row"]

    monkeypatch.setattr(diagrams, "get_diagrams", lambda: break_diagram(squash))
    with pytest.raises(DiagramLayoutError, match="too close"):
        draw_diagram("round3")


def test_figure_shows_as_png_without_a_matplotlib_backend() -> None:
    fig = draw_diagram("round1")
    assert isinstance(fig, DiagramFigure)
    assert fig._repr_png_().startswith(PNG_MAGIC)


REVIEW_GRAPH_EDGES = {
    ("__start__", "review_page:classify", "dashed"),
    ("review_page:classify", "review_page:attorney_review", "dashed"),
    ("review_page:classify", "review_page:redact", "dashed"),
    ("review_page:classify", "review_page:produce", "dashed"),
    ("review_page:classify", "review_page:withhold", "dashed"),
    ("review_page:redact", "review_page:attorney_review", "dashed"),
    ("review_page:attorney_review", "review_page:__end__", "solid"),
    ("review_page:produce", "review_page:__end__", "solid"),
    ("review_page:withhold", "review_page:__end__", "solid"),
    ("review_page:__end__", "__end__", "solid"),
}
REVIEW_GRAPH_CELLS = {
    "__start__": (0, 1.5),
    "review_page:classify": (1, 1.5),
    "review_page:attorney_review": (2, 0),
    "review_page:redact": (2, 1),
    "review_page:produce": (2, 2),
    "review_page:withhold": (2, 3),
    "review_page:__end__": (3, 1.5),
    "__end__": (4, 1.5),
}
REVIEW_GRAPH_STYLES = {
    "__start__": ("START\nbatch of pages", "note"),
    "review_page:classify": ("classify\nengine + code router", "step"),
    "review_page:attorney_review": ("attorney_review\ninterrupt()", "attorney"),
    "review_page:redact": ("redact", "redact"),
    "review_page:produce": ("produce", "produce"),
    "review_page:withhold": ("withhold", "withhold"),
    "review_page:__end__": ("page done", "note"),
    "__end__": ("END\nall results", "note"),
}
REVIEW_PAGE_NODES = {
    "review_page:classify",
    "review_page:attorney_review",
    "review_page:redact",
    "review_page:produce",
    "review_page:withhold",
    "review_page:__end__",
}


@pytest.fixture(scope="module")
def review_graph() -> Graph:
    classifier = StubClassifier(
        make_result(responsive=2.0, confidence=0.9, privileged=0.0, pii=0.0)
    )
    compiled = build_review_graph(classifier, FakeRedactor(), review_checkpointer())
    return compiled.get_graph(xray=True)


def test_graph_spec_lays_out_the_compiled_review_graph(review_graph: Graph) -> None:
    spec = graph_spec(review_graph)
    edges = [(e.source, e.target, e.style) for e in spec.edges]
    assert len(edges) == 10
    assert set(edges) == REVIEW_GRAPH_EDGES
    assert [e.label for e in spec.edges] == [""] * 10
    assert (spec.cols, spec.rows) == (5, 4)
    assert {n.id: (n.col, n.row) for n in spec.nodes} == REVIEW_GRAPH_CELLS


def test_graph_spec_labels_nodes_from_toml(review_graph: Graph) -> None:
    spec = graph_spec(review_graph)
    assert spec.title == "The review graph, drawn from the compiled LangGraph"
    assert {n.id: (n.label, n.kind) for n in spec.nodes} == REVIEW_GRAPH_STYLES
    (group,) = spec.groups
    assert group.label == "review_page: one subgraph per page, fanned out with Send"
    assert set(group.nodes) == REVIEW_PAGE_NODES


def test_draw_graph_draws_every_node_and_edge(review_graph: Graph) -> None:
    fig = draw_graph(review_graph)
    assert isinstance(fig, DiagramFigure)
    patches = fig.axes[0].patches
    assert {p.get_gid() for p in patches} >= set(REVIEW_GRAPH_CELLS)
    arrows = [p for p in patches if isinstance(p, FancyArrowPatch)]
    assert len(arrows) == 10
    assert [p.get_linestyle() for p in arrows].count("--") == 6


class TinyState(TypedDict):
    value: int


def keep(state: TinyState) -> TinyState:
    return state


def test_graph_node_missing_from_toml_is_rejected() -> None:
    tiny = StateGraph(TinyState)
    tiny.add_node("x", keep)
    tiny.add_edge(START, "x")
    tiny.add_edge("x", END)
    with pytest.raises(DiagramLayoutError, match=r"graph node 'x' has no"):
        graph_spec(tiny.compile().get_graph())


def test_graph_without_subgraph_nodes_has_no_group() -> None:
    graph = Graph()
    start, end = graph.add_node(None, "__start__"), graph.add_node(None, "__end__")
    graph.add_edge(start, end)
    spec = graph_spec(graph)
    assert (spec.cols, spec.rows) == (2, 1)
    assert spec.groups == ()


def test_unreachable_graph_node_is_rejected() -> None:
    graph = Graph()
    graph.add_node(None, "__start__")
    graph.add_node(None, "review_page:produce")
    with pytest.raises(
        DiagramLayoutError, match=r"\['review_page:produce'\] are not reachable"
    ):
        graph_spec(graph)


def test_unknown_graph_kind_is_rejected() -> None:
    data = copy.deepcopy(raw())
    data["graph"]["nodes"]["__start__"]["kind"] = "sparkle"
    with pytest.raises(ConfigError, match=r"\[graph\] uses unknown kinds"):
        parse_diagrams(data, get_settings())

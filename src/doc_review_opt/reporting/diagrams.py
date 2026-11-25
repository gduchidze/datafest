"""Flow diagrams drawn with matplotlib from diagrams.toml or a compiled graph."""

from __future__ import annotations

from io import BytesIO
from itertools import combinations

from langchain_core.runnables.graph import Graph
from langgraph.graph import START
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from doc_review_opt.config import (
    DiagramEdge,
    DiagramGroup,
    DiagramLayout,
    DiagramNode,
    DiagramSpec,
    get_diagrams,
    get_settings,
)
from doc_review_opt.exceptions import DiagramLayoutError

PALETTE = get_settings().palette
CHART_WIDTH = get_settings().chart.width
TITLE_Y = 0.35
SUBGRAPH_SEPARATOR = ":"

Point = tuple[float, float]
Size = tuple[float, float]


class DiagramFigure(Figure):
    def _repr_png_(self) -> bytes:
        buffer = BytesIO()
        self.savefig(buffer, format="png", bbox_inches="tight")
        return buffer.getvalue()


def list_diagrams() -> list[str]:
    return list(get_diagrams().diagrams)


def _text_size(axis: Axes, renderer: RendererBase, label: str, size: float) -> Size:
    probe = axis.text(0, 0, label, fontsize=size, ha="center", va="center")
    box = probe.get_window_extent(renderer)
    probe.remove()
    dpi = axis.figure.dpi
    return box.width / dpi, box.height / dpi


def _box_sizes(
    axis: Axes, renderer: RendererBase, spec: DiagramSpec, layout: DiagramLayout
) -> dict[str, Size]:
    texts = {
        node.id: _text_size(axis, renderer, node.label, layout.font_size)
        for node in spec.nodes
    }
    widest: dict[float, float] = {}
    tallest: dict[float, float] = {}
    for node in spec.nodes:
        width, height = texts[node.id]
        widest[node.col] = max(widest.get(node.col, 0.0), width)
        tallest[node.row] = max(tallest.get(node.row, 0.0), height)
    return {
        node.id: (
            widest[node.col] + 2 * layout.pad_x,
            tallest[node.row] + 2 * layout.pad_y,
        )
        for node in spec.nodes
    }


def _center(node: DiagramNode, spec: DiagramSpec, layout: DiagramLayout) -> Point:
    x = (node.col + 0.5) * CHART_WIDTH / spec.cols
    y = layout.top_margin + (node.row + 0.5) * layout.row_height
    return x, y


def _edge_end(center: Point, size: Size, toward: Point) -> Point:
    dx, dy = toward[0] - center[0], toward[1] - center[1]
    scale = min(
        (size[0] / 2) / abs(dx) if dx else float("inf"),
        (size[1] / 2) / abs(dy) if dy else float("inf"),
    )
    return center[0] + dx * scale, center[1] + dy * scale


def _check_spacing(
    name: str, centers: dict[str, Point], sizes: dict[str, Size], gap: float
) -> None:
    for first, second in combinations(centers, 2):
        reach_x = (sizes[first][0] + sizes[second][0]) / 2 + gap
        reach_y = (sizes[first][1] + sizes[second][1]) / 2
        near_x = abs(centers[first][0] - centers[second][0]) < reach_x
        near_y = abs(centers[first][1] - centers[second][1]) < reach_y
        if near_x and near_y:
            raise DiagramLayoutError(f"{name}: {first} and {second} are too close")


def _draw_group(
    axis: Axes,
    group: DiagramGroup,
    centers: dict[str, Point],
    sizes: dict[str, Size],
    layout: DiagramLayout,
) -> None:
    lefts = [centers[n][0] - sizes[n][0] / 2 for n in group.nodes]
    rights = [centers[n][0] + sizes[n][0] / 2 for n in group.nodes]
    tops = [centers[n][1] - sizes[n][1] / 2 for n in group.nodes]
    bottoms = [centers[n][1] + sizes[n][1] / 2 for n in group.nodes]
    pad = layout.group_pad
    left = min(lefts) - pad
    top = min(tops) - pad - layout.group_label_pad
    width = max(rights) - min(lefts) + 2 * pad
    height = max(bottoms) - min(tops) + 2 * pad + layout.group_label_pad
    axis.add_patch(
        Rectangle(
            (left, top),
            width,
            height,
            fc="none",
            ec=PALETTE.text,
            lw=layout.line_width,
            ls="--",
            zorder=0,
        )
    )
    axis.text(
        left + layout.group_pad / 2,
        top + layout.group_label_pad / 2 + layout.group_pad / 4,
        group.label,
        fontsize=layout.label_font_size,
        fontweight="bold",
        color=PALETTE.text,
        ha="left",
        va="center",
    )


def _draw_node(
    axis: Axes, node: DiagramNode, center: Point, size: Size, layout: DiagramLayout
) -> FancyBboxPatch:
    style = get_diagrams().kinds[node.kind]
    patch = FancyBboxPatch(
        (center[0] - size[0] / 2, center[1] - size[1] / 2),
        size[0],
        size[1],
        boxstyle=f"round,pad=0,rounding_size={layout.corner}",
        fc=getattr(PALETTE, style.fill),
        ec=getattr(PALETTE, style.edge) if style.edge else PALETTE.text,
        lw=layout.model_line_width if style.edge else layout.line_width,
        zorder=2,
        gid=node.id,
    )
    axis.add_patch(patch)
    axis.text(
        *center,
        node.label,
        fontsize=layout.font_size,
        color=getattr(PALETTE, style.text),
        ha="center",
        va="center",
        zorder=3,
        gid=node.id,
    )
    return patch


def _draw_edge(
    axis: Axes,
    edge: DiagramEdge,
    centers: dict[str, Point],
    patches: dict[str, FancyBboxPatch],
    sizes: dict[str, Size],
    layout: DiagramLayout,
) -> None:
    start, end = centers[edge.source], centers[edge.target]
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            patchA=patches[edge.source],
            patchB=patches[edge.target],
            arrowstyle="-|>",
            mutation_scale=layout.arrow_scale,
            shrinkA=2,
            shrinkB=2,
            color=PALETTE.text,
            lw=layout.line_width,
            ls="--" if edge.style == "dashed" else "-",
            zorder=1,
        )
    )
    if not edge.label:
        return
    a = _edge_end(start, sizes[edge.source], end)
    b = _edge_end(end, sizes[edge.target], start)
    axis.text(
        a[0] + (b[0] - a[0]) * edge.at,
        a[1] + (b[1] - a[1]) * edge.at,
        edge.label,
        fontsize=layout.label_font_size,
        color=PALETTE.text,
        ha="center",
        va="center",
        zorder=4,
        gid=f"{edge.source}->{edge.target}",
        bbox={"boxstyle": "round,pad=0.2", "fc": PALETTE.surface, "ec": "none"},
    )


def _render(name: str, spec: DiagramSpec) -> Figure:
    layout = get_diagrams().layout
    height = layout.top_margin + spec.rows * layout.row_height + layout.bottom_margin
    fig = DiagramFigure(figsize=(CHART_WIDTH, height), facecolor=PALETTE.surface)
    renderer = FigureCanvasAgg(fig).get_renderer()  # type: ignore[no-untyped-call]
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, CHART_WIDTH)
    axis.set_ylim(height, 0)
    axis.axis("off")
    sizes = _box_sizes(axis, renderer, spec, layout)
    centers = {node.id: _center(node, spec, layout) for node in spec.nodes}
    _check_spacing(name, centers, sizes, layout.min_gap)
    axis.text(
        0.15,
        TITLE_Y,
        spec.title,
        fontsize=layout.title_size,
        fontweight="bold",
        color=PALETTE.text,
        ha="left",
        va="center",
    )
    for group in spec.groups:
        _draw_group(axis, group, centers, sizes, layout)
    patches = {
        node.id: _draw_node(axis, node, centers[node.id], sizes[node.id], layout)
        for node in spec.nodes
    }
    for edge in spec.edges:
        _draw_edge(axis, edge, centers, patches, sizes, layout)
    return fig


def draw_diagram(name: str) -> Figure:
    return _render(name, get_diagrams().diagrams[name])


def _depths(graph: Graph) -> dict[str, int]:
    depths = {START: 0}
    frontier = {START}
    level = 0
    while frontier:
        level += 1
        frontier = {e.target for e in graph.edges if e.source in frontier}
        frontier -= depths.keys()
        depths |= dict.fromkeys(frontier, level)
    unreachable = graph.nodes.keys() - depths.keys()
    if unreachable:
        raise DiagramLayoutError(
            f"graph nodes {sorted(unreachable)} are not reachable from {START}"
        )
    return depths


def _columns(graph: Graph) -> list[list[str]]:
    depths = _depths(graph)
    columns: list[list[str]] = [[] for _ in range(max(depths.values()) + 1)]
    for node_id in graph.nodes:
        columns[depths[node_id]].append(node_id)
    return columns


def _graph_node(node_id: str, col: int, row: float) -> DiagramNode:
    style = get_diagrams().graph.nodes.get(node_id)
    if style is None:
        raise DiagramLayoutError(
            f"graph node {node_id!r} has no [graph.nodes] entry in diagrams.toml"
        )
    return DiagramNode(id=node_id, label=style.label, kind=style.kind, col=col, row=row)


def graph_spec(graph: Graph) -> DiagramSpec:
    style = get_diagrams().graph
    columns = _columns(graph)
    rows = max(len(ids) for ids in columns)
    nodes = tuple(
        _graph_node(node_id, col, row + (rows - len(ids)) / 2)
        for col, ids in enumerate(columns)
        for row, node_id in enumerate(ids)
    )
    edges = tuple(
        DiagramEdge.model_validate(
            {
                "from": e.source,
                "to": e.target,
                "style": "dashed" if e.conditional else "solid",
            }
        )
        for e in graph.edges
    )
    grouped = tuple(node_id for node_id in graph.nodes if SUBGRAPH_SEPARATOR in node_id)
    groups = (DiagramGroup(label=style.group, nodes=grouped),) if grouped else ()
    return DiagramSpec(
        title=style.title,
        cols=len(columns),
        rows=rows,
        nodes=nodes,
        edges=edges,
        groups=groups,
    )


def draw_graph(graph: Graph) -> Figure:
    spec = graph_spec(graph)
    return _render(spec.title, spec)

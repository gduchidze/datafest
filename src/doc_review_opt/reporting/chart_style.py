"""Shared chart colors and axis styling, all read from config.toml."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend

from doc_review_opt.config import get_settings

SETTINGS = get_settings()
PALETTE = SETTINGS.palette
CHART = SETTINGS.chart
ACCEPTED_TEXT = "✓ ACCEPTED"
REJECTED_TEXT = "✗ REJECTED"
ROUTE_LABELS = {
    "produce": "produce",
    "withhold": "withhold",
    "redact": "redact",
    "attorney_review": "attorney review",
}


def color(entity: str) -> str:
    return str(getattr(PALETTE, SETTINGS.entity_colors[entity]))


def signed_percent(value: float) -> str:
    text = f"{value:+.0%}"
    return "0%" if text in ("+0%", "-0%") else text


def status_color(accepted: bool) -> str:
    return PALETTE.good if accepted else PALETTE.critical


def status_text(accepted: bool) -> str:
    return ACCEPTED_TEXT if accepted else REJECTED_TEXT


def new_figure(
    title: str, columns: int = 1, height: float = CHART.height
) -> tuple[Figure, list[Axes]]:
    fig = Figure(
        figsize=(CHART.width, height),
        facecolor=PALETTE.surface,
        layout="constrained",
    )
    fig.suptitle(
        title, fontsize=CHART.title_size, fontweight="bold", color=PALETTE.text
    )
    axes = [
        fig.add_subplot(1, columns, index + 1, facecolor=PALETTE.surface)
        for index in range(columns)
    ]
    return fig, axes


def style_axis(
    axis: Axes,
    *,
    ylabel: str = "",
    xlabel: str = "",
    title: str = "",
    grid_axis: Literal["x", "y"] = "y",
) -> None:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(PALETTE.grid)
    axis.set_axisbelow(True)
    axis.grid(
        axis=grid_axis, color=PALETTE.grid, linewidth=CHART.grid_width, linestyle="-"
    )
    axis.tick_params(colors=PALETTE.text, labelsize=CHART.font_size)
    axis.set_ylabel(ylabel, fontsize=CHART.font_size, color=PALETTE.text)
    axis.set_xlabel(xlabel, fontsize=CHART.font_size, color=PALETTE.text)
    if title:
        axis.set_title(title, fontsize=CHART.font_size, color=PALETTE.text, loc="left")


def label_bars(
    axis: Axes, positions: Sequence[float], tops: Sequence[float], texts: Sequence[str]
) -> None:
    for position, top, text in zip(positions, tops, texts, strict=True):
        axis.annotate(
            text,
            (position, top),
            xytext=(0, CHART.label_pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=CHART.font_size,
            color=PALETTE.text,
        )


def _label_side(
    values: Sequence[float], index: int
) -> tuple[bool, Literal["left", "center", "right"]]:
    value = values[index]
    before = values[index - 1] if index > 0 else None
    after = values[index + 1] if index + 1 < len(values) else None
    if before is None:
        return (after is not None and after > value), "center"
    if after is None:
        return False, "left" if before > value else "center"
    if before > value and after > value:
        return True, "center"
    if before < value < after:
        return False, "right"
    if before > value > after:
        return False, "left"
    return False, "center"


def label_points(
    axis: Axes,
    positions: Sequence[float],
    values: Sequence[float],
    texts: Sequence[str],
) -> None:
    for index, (position, value, text) in enumerate(
        zip(positions, values, texts, strict=True)
    ):
        below, align = _label_side(values, index)
        shift = {"left": CHART.label_pad, "center": 0.0, "right": -CHART.label_pad}
        axis.annotate(
            text,
            (position, value),
            xytext=(
                shift[align],
                -CHART.label_pad * 2 if below else CHART.label_pad * 2,
            ),
            textcoords="offset points",
            ha=align,
            va="top" if below else "bottom",
            fontsize=CHART.font_size,
            color=PALETTE.text,
            bbox={
                "boxstyle": f"square,pad={CHART.label_box_pad}",
                "fc": PALETTE.surface,
                "ec": "none",
            },
            zorder=5,
        )


def label_hbars(
    axis: Axes, positions: Sequence[float], ends: Sequence[float], texts: Sequence[str]
) -> None:
    for position, end, text in zip(positions, ends, texts, strict=True):
        axis.annotate(
            text,
            (end, position),
            xytext=(CHART.label_pad, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=CHART.font_size,
            color=PALETTE.text,
        )


def _recolor(legend: Legend) -> None:
    for text in legend.get_texts():
        text.set_color(PALETTE.text)


def add_legend(
    axis: Axes, loc: Literal["upper right", "upper left"] = "upper right"
) -> None:
    _recolor(axis.legend(frameon=False, fontsize=CHART.font_size, loc=loc))


def add_legend_above(axis: Axes, columns: int) -> None:
    _recolor(
        axis.legend(
            frameon=False,
            fontsize=CHART.font_size,
            ncols=columns,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
        )
    )

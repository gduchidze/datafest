"""Per-variant summaries, round-by-round deltas, per-page views and the round charts."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from doc_review_opt.agent.variants import BASELINE
from doc_review_opt.config import get_settings
from doc_review_opt.reporting.chart_style import (
    CHART,
    PALETTE,
    ROUTE_LABELS,
    add_legend,
    add_legend_above,
    color,
    label_bars,
    label_hbars,
    label_points,
    new_figure,
    signed_percent,
    status_color,
    status_text,
    style_axis,
)

MONTHLY_PAGES = get_settings().scoring.monthly_pages
LIVE_RUN = get_settings().live_run
DIFFICULTY_ORDER = ("simple", "routine", "borderline", "trap")
COST_COLUMN = "model_cost_per_page_usd"
LATENCY_COLUMN = "p50_latency_s"
DELTA_COLUMNS = [
    "mean_sync_tokens",
    "mean_total_tokens",
    "p50_latency_s",
    "model_cost_per_page_usd",
    "cost_per_page_usd",
]
IMPACT_COLUMNS = [
    "variant_label",
    "mean_quality",
    "policy_compliance",
    "escalation_rate",
    "mean_tool_calls",
    "mean_extra_tool_calls",
    "mean_sync_tokens",
    "mean_total_tokens",
    "mean_cached_tokens",
    "p50_latency_s",
    "model_cost_per_page_usd",
    "cost_per_page_usd",
]


def summarize(traces: pd.DataFrame) -> pd.DataFrame:
    grouped = traces.groupby(["variant", "variant_label"], sort=False)
    summary = grouped.agg(
        pages=("doc_id", "count"),
        mean_quality=("quality_score", "mean"),
        policy_compliance=("policy_compliant", "mean"),
        route_accuracy=("route_correct", "mean"),
        escalation_accuracy=("escalation_correct", "mean"),
        escalation_rate=("escalated", "mean"),
        privileged_leaks=("privileged_leak", "sum"),
        pii_leaks=("pii_leak", "sum"),
        mean_tool_calls=("tool_calls", "mean"),
        mean_extra_tool_calls=("extra_tool_calls", "mean"),
        mean_input_tokens=("input_tokens", "mean"),
        mean_cached_tokens=("cached_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"),
        mean_sync_tokens=("sync_tokens", "mean"),
        mean_total_tokens=("total_tokens", "mean"),
        p50_latency_s=("latency_s", "median"),
        p95_latency_s=("latency_s", lambda s: s.quantile(0.95)),
        sync_cost_per_page_usd=("sync_cost_usd", "mean"),
        background_cost_per_page_usd=("background_cost_usd", "mean"),
        model_cost_per_page_usd=("model_cost_usd", "mean"),
        attorney_cost_per_page_usd=("attorney_cost_usd", "mean"),
        cost_per_page_usd=("cost_usd", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        correct_pages=("route_correct", "sum"),
    ).reset_index()
    summary["blended_cost_per_verified_page"] = summary["total_cost_usd"] / summary[
        "correct_pages"
    ].where(summary["correct_pages"] > 0)
    base = summary[summary["variant"] == BASELINE]
    base_row = base.iloc[0] if len(base) else summary.iloc[0]
    for column, source in (
        ("cost_reduction_vs_baseline", "model_cost_per_page_usd"),
        ("token_reduction_vs_baseline", "mean_total_tokens"),
        ("latency_reduction_vs_baseline", "p50_latency_s"),
    ):
        summary[column] = 1 - summary[source] / base_row[source]
    summary["monthly_cost_at_100k_pages"] = summary["cost_per_page_usd"] * MONTHLY_PAGES
    summary["monthly_model_cost_at_100k_pages"] = (
        summary["model_cost_per_page_usd"] * MONTHLY_PAGES
    )
    return summary


def round_impact(summary: pd.DataFrame) -> pd.DataFrame:
    impact = summary[IMPACT_COLUMNS].copy()
    for column in DELTA_COLUMNS:
        impact[f"{column}_delta_vs_previous"] = impact[column].diff()
    impact["quality_delta_vs_previous"] = impact["mean_quality"].diff()
    return impact


def per_page_view(
    traces: pd.DataFrame, variants: Sequence[str], columns: Sequence[str]
) -> pd.DataFrame:
    view = traces[traces["variant"].isin(variants)]
    return view[["doc_id", "variant_label", *columns]].reset_index(drop=True)


def recorded_live_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": row.variant,
            "mean_quality": row.quality,
            "escalation_rate": row.escalation_rate,
            "mean_sync_tokens": row.sync_tokens,
            LATENCY_COLUMN: row.p50_latency_s,
            COST_COLUMN: row.model_cost_per_page_usd,
        }
        for row in LIVE_RUN.rounds
    )


def recorded_live_note() -> str:
    return f"live: {LIVE_RUN.pages} pages, {LIVE_RUN.date}"


def _short_labels(summary: pd.DataFrame) -> list[str]:
    return list(summary["variant_label"].str.replace("Round ", "R", regex=False))


def _floats(values: pd.Series) -> list[float]:
    return [float(value) for value in values]


def _log_limits(axis: Axes, values: Sequence[float]) -> None:
    axis.set_yscale("log")
    axis.set_ylim(
        min(values) * CHART.log_floor_factor, max(values) * CHART.log_headroom
    )


def _set_rounds(axis: Axes, labels: Sequence[str], count: int | None = None) -> None:
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_xlim(-0.5, (count or len(labels)) - 0.5)


def _bars(
    axis: Axes, labels: Sequence[str], values: Sequence[float], fill: str
) -> list[float]:
    positions = [float(i) for i in range(len(labels))]
    axis.bar(positions, values, width=CHART.bar_width, color=fill, zorder=3)
    _set_rounds(axis, labels)
    return positions


def _change(values: Sequence[float]) -> float:
    return values[-1] / values[0] - 1


def plot_round_chart(summary: pd.DataFrame, cost_column: str = COST_COLUMN) -> Figure:
    labels = _short_labels(summary)
    tokens = _floats(summary["mean_sync_tokens"])
    costs = _floats(summary[cost_column])
    quality = _floats(summary["mean_quality"])
    fig, axes = new_figure(
        f"Sync tokens {signed_percent(_change(tokens))} and model cost "
        f"{signed_percent(_change(costs))} "
        f"from {labels[0]} to {labels[-1]}; "
        f"quality {quality[0]:.2f} to {quality[-1]:.2f}",
        columns=3,
    )
    positions = _bars(axes[0], labels, tokens, color("sync_tokens"))
    label_bars(axes[0], positions, tokens, [f"{v:,.0f}" for v in tokens])
    axes[0].margins(y=0.15)
    style_axis(axes[0], ylabel="tokens per page", title="Sync tokens (lower is better)")
    positions = _bars(axes[1], labels, costs, color("model_cost"))
    _log_limits(axes[1], costs)
    style_axis(
        axes[1],
        ylabel="USD per page (log scale)",
        title="Model cost (lower is better)",
    )
    axes[2].plot(
        positions,
        quality,
        marker="o",
        color=color("quality"),
        linewidth=CHART.line_width,
        markersize=CHART.marker_size,
        zorder=3,
    )
    label_points(axes[2], positions, quality, [f"{v:.2f}" for v in quality])
    axes[2].set_ylim(0, 1.15)
    _set_rounds(axes[2], labels)
    style_axis(axes[2], ylabel="score (0 to 1)", title="Quality (higher is better)")
    return fig


def plot_progress(
    summary: pd.DataFrame, upto_variant: str, cost_column: str = COST_COLUMN
) -> Figure:
    last = summary.index[summary["variant"] == upto_variant][0]
    shown = summary.loc[:last]
    labels = _short_labels(shown)
    total = len(summary)
    costs = _floats(shown[cost_column])
    latency = _floats(shown[LATENCY_COLUMN])
    quality = _floats(shown["mean_quality"])
    fig, axes = new_figure(
        f"Through {labels[-1]}, cost per page {signed_percent(_change(costs))}, "
        f"p50 latency {signed_percent(_change(latency))}, "
        f"quality {quality[-1] - quality[0]:+.2f}",
        columns=3,
        height=CHART.height * 0.8,
    )
    for axis, values, entity, title, fmt in (
        (
            axes[0],
            costs,
            "model_cost",
            "Cost per page (USD, lower is better)",
            "",
        ),
        (axes[1], latency, "latency", "p50 latency (s, lower is better)", "{:.2f}"),
        (axes[2], quality, "quality", "Quality score (higher is better)", "{:.2f}"),
    ):
        positions = [float(i) for i in range(len(labels))]
        axis.plot(
            positions,
            values,
            marker="o",
            color=color(entity),
            linewidth=CHART.line_width,
            markersize=CHART.marker_size,
            zorder=3,
        )
        if fmt:
            label_points(axis, positions, values, [fmt.format(v) for v in values])
        _set_rounds(axis, labels, total)
        style_axis(axis, title=title)
    _log_limits(axes[0], costs)
    axes[1].set_ylim(0, max(latency) * 1.25)
    axes[2].set_ylim(0, 1.15)
    return fig


def plot_eval_set(eval_table: pd.DataFrame) -> Figure:
    routes = [r for r in ROUTE_LABELS if r in set(eval_table["expected_route"])]
    route_counts = [int((eval_table["expected_route"] == r).sum()) for r in routes]
    levels = [d for d in DIFFICULTY_ORDER if d in set(eval_table["difficulty"])]
    level_counts = [int((eval_table["difficulty"] == d).sum()) for d in levels]
    hard = sum(
        count
        for level, count in zip(levels, level_counts, strict=True)
        if level in ("borderline", "trap")
    )
    fig, axes = new_figure(
        f"{len(eval_table)} pages; {hard} are borderline or trap pages",
        columns=2,
        height=CHART.height * 0.7,
    )
    for axis, names, counts, fills, title in (
        (
            axes[0],
            [ROUTE_LABELS[r] for r in routes],
            route_counts,
            [color(r) for r in routes],
            "Pages by expected route",
        ),
        (
            axes[1],
            levels,
            level_counts,
            [color("difficulty")] * len(levels),
            "Pages by difficulty",
        ),
    ):
        positions = [float(i) for i in range(len(names))]
        axis.barh(positions, counts, height=CHART.bar_width, color=fills, zorder=3)
        axis.set_yticks(positions)
        axis.set_yticklabels(names)
        axis.invert_yaxis()
        label_hbars(
            axis, positions, [float(c) for c in counts], [str(c) for c in counts]
        )
        axis.margins(x=0.15)
        style_axis(axis, xlabel="pages", title=title, grid_axis="x")
    return fig


def plot_cost_latency(summary: pd.DataFrame) -> Figure:
    labels = _short_labels(summary)
    costs = _floats(summary[COST_COLUMN])
    latency = _floats(summary[LATENCY_COLUMN])
    fig, axes = new_figure(
        f"Model cost per page {signed_percent(_change(costs))} and p50 latency "
        f"{signed_percent(_change(latency))} from {labels[0]} to {labels[-1]} "
        "(simulated)",
        columns=2,
    )
    positions = _bars(axes[0], labels, costs, color("model_cost"))
    _log_limits(axes[0], costs)
    texts = [f"${costs[0]:.5f}"] + [
        f"${now:.5f}\n{signed_percent(now / before - 1)}"
        for before, now in itertools.pairwise(costs)
    ]
    label_bars(axes[0], positions, costs, texts)
    style_axis(
        axes[0],
        ylabel="USD per page (log scale)",
        title="Model cost per page, % vs previous round (lower is better)",
    )
    _bars(axes[1], labels, latency, color("latency"))
    label_bars(axes[1], positions, latency, [f"{v:.2f}s" for v in latency])
    axes[1].margins(y=0.15)
    style_axis(axes[1], ylabel="seconds", title="p50 latency (lower is better)")
    return fig


def plot_cost_split(summary: pd.DataFrame) -> Figure:
    labels = _short_labels(summary)
    model = _floats(summary["model_cost_per_page_usd"])
    attorney = _floats(summary["attorney_cost_per_page_usd"])
    totals = [m + a for m, a in zip(model, attorney, strict=True)]
    fig, (axis,) = new_figure(
        f"Model calls are only {model[-1] / totals[-1]:.1%} of the cost per page "
        f"in {labels[-1]}; attorney time is the rest",
    )
    positions = [float(i) for i in range(len(labels))]
    axis.bar(
        positions,
        model,
        width=CHART.bar_width,
        color=color("model_cost"),
        label="Model cost",
        zorder=3,
    )
    axis.bar(
        positions,
        attorney,
        bottom=model,
        width=CHART.bar_width,
        color=color("attorney_cost"),
        label="Attorney cost",
        zorder=3,
    )
    label_bars(axis, positions, totals, [f"${t:.2f}" for t in totals])
    _set_rounds(axis, labels)
    axis.margins(y=0.15)
    style_axis(axis, ylabel="USD per page (lower is better)")
    add_legend(axis)
    return fig


def plot_quality_gate(gates: pd.DataFrame, floor: float) -> Figure:
    labels = list(gates["label"])
    rates = _floats(gates["pass_rate"])
    accepted = [bool(a) for a in gates["accepted"]]
    fig, (axis,) = new_figure(
        f"{sum(accepted)} of {len(accepted)} versions are accepted: "
        f"every one needs at least {floor:.0%} guardrail passes and zero leaks",
        height=CHART.height * 1.1,
    )
    positions = [float(i) for i in range(len(labels))]
    axis.barh(
        positions,
        rates,
        height=CHART.bar_width,
        color=[status_color(a) for a in accepted],
        zorder=3,
    )
    axis.axvline(
        floor, color=PALETTE.text, linestyle="--", linewidth=CHART.line_width, zorder=4
    )
    axis.annotate(
        f"{floor:.0%} floor",
        (floor, -0.7),
        xytext=(-CHART.label_pad, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=CHART.font_size,
        color=PALETTE.text,
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_ylim(len(labels) - 0.4, -1.2)
    axis.set_xlim(0, 1.35)
    axis.set_xticks(np.linspace(0, 1, 5))
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    label_hbars(
        axis,
        positions,
        [1.0] * len(rates),
        [f"{r:.0%}  {status_text(a)}" for r, a in zip(rates, accepted, strict=True)],
    )
    style_axis(axis, xlabel="guardrail pass rate (higher is better)", grid_axis="x")
    return fig


def plot_route_mix(frame: pd.DataFrame, variants: Sequence[str]) -> Figure:
    routes = list(ROUTE_LABELS)
    names = frame.drop_duplicates("variant").set_index("variant")["variant_label"]
    first = frame[frame["variant"] == variants[0]]
    rows = {"Expected (labels)": first["expected_route"]}
    rows |= {
        str(names[v]).replace("Round ", "R"): frame[frame["variant"] == v]["route"]
        for v in variants
    }
    shares = {
        row: [float((values == route).mean()) for route in routes]
        for row, values in rows.items()
    }
    review = routes.index("attorney_review")
    fig, (axis,) = new_figure(
        f"Attorney review share: {list(shares.values())[1][review]:.0%} in the "
        f"first version, {list(shares.values())[-1][review]:.0%} in the final round, "
        f"{shares['Expected (labels)'][review]:.0%} expected by the labels",
        height=CHART.height * 1.1,
    )
    positions = [float(i) for i in range(len(shares))]
    left = [0.0] * len(shares)
    for index, route in enumerate(routes):
        widths = [row[index] for row in shares.values()]
        axis.barh(
            positions,
            widths,
            left=left,
            height=CHART.bar_width,
            color=color(route),
            label=ROUTE_LABELS[route],
            zorder=3,
        )
        for position, start, width in zip(positions, left, widths, strict=True):
            if width >= CHART.min_label_share:
                axis.text(
                    start + width / 2,
                    position,
                    f"{width:.0%}",
                    ha="center",
                    va="center",
                    fontsize=CHART.font_size,
                    color=PALETTE.ink,
                )
        left = [start + width for start, width in zip(left, widths, strict=True)]
    axis.set_yticks(positions)
    axis.set_yticklabels(list(shares))
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    style_axis(axis, xlabel=f"share of the {len(first)} pages", grid_axis="x")
    add_legend_above(axis, len(routes))
    return fig


def plot_scorecard(frame: pd.DataFrame, variants: Sequence[str]) -> Figure:
    pages = (
        frame[frame["variant"] == variants[0]]
        .sort_values(["expected_route", "doc_id"])[["doc_id", "expected_route"]]
        .reset_index(drop=True)
    )
    row_of = {doc: i for i, doc in enumerate(pages["doc_id"])}
    names = frame.drop_duplicates("variant").set_index("variant")["variant_label"]
    points: dict[str, tuple[list[float], list[float]]] = {
        kind: ([], []) for kind in ("correct", "wrong", "leak")
    }
    for column, variant in enumerate(variants):
        for trace in frame[frame["variant"] == variant].itertuples():
            leak = bool(trace.privileged_leak) or bool(trace.pii_leak)
            kind = "leak" if leak else "correct" if trace.route_correct else "wrong"
            points[kind][0].append(float(column))
            points[kind][1].append(float(row_of[str(trace.doc_id)]))
    first, last = variants[0], variants[-1]

    def short_name(variant: str) -> str:
        return str(names[variant]).replace("Round ", "R")

    def errors(variant: str) -> int:
        view = frame[frame["variant"] == variant]
        return int((~view["route_correct"]).sum())

    fig, (axis,) = new_figure(
        f"Wrong routes fall from {errors(first)} in {short_name(first)} "
        f"to {errors(last)} in {short_name(last)} ({len(pages)} pages)",
        height=CHART.tall_height,
    )
    styles = {
        "correct": ("o", PALETTE.good, "✓ correct route"),
        "wrong": ("X", PALETTE.critical, "✗ wrong route"),
        "leak": ("^", PALETTE.critical, "⚠ leak: privileged page produced or PII left"),
    }
    for kind, (marker, fill, text) in styles.items():
        axis.scatter(
            points[kind][0],
            points[kind][1],
            marker=marker,
            s=CHART.marker_size**2,
            color=fill,
            label=text,
            zorder=3,
        )
    axis.set_yticks(range(len(pages)))
    axis.set_yticklabels(
        [
            f"{d}  ({ROUTE_LABELS[r]})"
            for d, r in zip(pages["doc_id"], pages["expected_route"], strict=True)
        ]
    )
    axis.set_xticks(range(len(variants)))
    axis.set_xticklabels(
        [short_name(v) for v in variants],
        rotation=20,
        ha="right",
    )
    axis.set_xlim(-0.5, len(variants) - 0.5)
    axis.invert_yaxis()
    axis.set_ylim(len(pages) - 0.5, -0.5)
    style_axis(axis, ylabel="page (expected route)")
    add_legend_above(axis, 3)
    return fig


def plot_sim_vs_live(summary: pd.DataFrame, live: pd.DataFrame, note: str) -> Figure:
    columns = ["variant", COST_COLUMN, LATENCY_COLUMN]
    merged = summary[columns].merge(
        live[columns], on="variant", how="left", suffixes=("_sim", "_live")
    )
    labels = _short_labels(summary)
    shown = merged.dropna()
    fig, axes = new_figure(
        f"Simulated vs live ({note}): live p50 latency "
        f"{signed_percent(_change(_floats(shown[LATENCY_COLUMN + '_live'])))} "
        f"from {labels[0]} to {labels[len(shown) - 1]}",
        columns=2,
    )
    width = CHART.bar_width / 2
    positions = np.arange(len(labels))
    for axis, column, ylabel, title in (
        (
            axes[0],
            COST_COLUMN,
            "USD per page (log scale)",
            "Model cost per page (lower is better)",
        ),
        (axes[1], LATENCY_COLUMN, "seconds", "p50 latency (lower is better)"),
    ):
        both: list[float] = []
        for offset, side, entity in (
            (-width / 2, "sim", "simulated"),
            (width / 2, "live", "live"),
        ):
            values = _floats(merged[f"{column}_{side}"])
            axis.bar(
                positions + offset,
                values,
                width=width,
                color=color(entity),
                label=entity,
                zorder=3,
            )
            both += [v for v in values if v == v]
        if column == COST_COLUMN:
            _log_limits(axis, both)
        _set_rounds(axis, labels)
        style_axis(axis, ylabel=ylabel, title=title)
    add_legend(axes[0])
    return fig

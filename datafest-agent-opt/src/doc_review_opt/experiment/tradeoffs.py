"""Decisions versus LLM: step matrix, engine comparison, sweep, cascade, calibration."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from openai import AsyncOpenAI

from doc_review_opt.agent.classifiers.decisions import DecisionsClassifier
from doc_review_opt.agent.classifiers.llm_probability import LLMProbabilityClassifier
from doc_review_opt.agent.classifiers.tool_agent import LiveAgentClassifier
from doc_review_opt.agent.prompts import REDACTION_PROMPT
from doc_review_opt.agent.protocols import Classifier
from doc_review_opt.agent.redactor import OpenAIRedactor
from doc_review_opt.agent.routing import (
    DEFAULT_POLICY,
    RoutingPolicy,
    decide,
    flags_pii,
)
from doc_review_opt.agent.variants import (
    CALIBRATED_POLICY,
    CONTROLS,
    DECISION_MODEL,
    ROUTER_ONLY,
    VARIANTS,
)
from doc_review_opt.config import Calibration, get_settings
from doc_review_opt.exceptions import BudgetExceededError
from doc_review_opt.experiment.live import LIVE_PAGE_LIMIT, BudgetTracker, run_traced
from doc_review_opt.experiment.runs import (
    ReviewRun,
    review_pages,
    scripted_attorney_decisions,
)
from doc_review_opt.experiment.simulation import (
    SimulatedAgentClassifier,
    SimulatedRedactor,
    SimulatedTypedClassifier,
    scrub_pii,
    simulate_variant,
)
from doc_review_opt.experiment.traces import (
    ATTORNEY_REVIEW_COST_USD_PER_PAGE,
    PageOutcome,
    Trace,
    evaluate_trace_quality,
    traces_frame,
)
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.models import ClassifierResult, ReviewPage, TokenUsage
from doc_review_opt.pricing import approx_tokens, estimate_cost_usd
from doc_review_opt.reporting.chart_style import (
    CHART,
    PALETTE,
    add_legend,
    color,
    label_bars,
    new_figure,
    style_axis,
)

logger = logging.getLogger(__name__)

SETTINGS = get_settings()
LIVE = SETTINGS.live
MODELS = SETTINGS.models
MONTHLY_PAGES = SETTINGS.scoring.monthly_pages
ENGINE_LABELS = {
    "decisions": f"Decisions ({MODELS.decisions})",
    "llm-luna": f"LLM probabilities ({MODELS.luna})",
    "llm-terra": f"LLM probabilities ({MODELS.terra})",
    "agent": f"Agent ({MODELS.terra})",
}
ENGINE_SHORT = {
    "decisions": "Decisions",
    "llm-luna": f"{MODELS.luna} LLM",
    "llm-terra": f"{MODELS.terra} LLM",
    "agent": f"{MODELS.terra} agent",
}
TYPED_ENGINES = ("decisions", "llm-luna", "llm-terra")

STEP_MATRIX = [
    {
        "step": "Responsiveness rubric",
        "fits": "Decisions",
        "why": "A fixed 0-2 rubric is a typed score: one cheap call, no text to write.",
    },
    {
        "step": "Privilege predicate",
        "fits": "Decisions",
        "why": "A yes/no with a probability feeds a low escalation threshold; the "
        "cascade sends the ambiguous band to a stronger model.",
    },
    {
        "step": "PII presence",
        "fits": "Decisions",
        "why": "Deciding whether personal data is present is a predicate; finding "
        "the spans is not (Decisions cannot extract to your own schema).",
    },
    {
        "step": "Near-duplicate check",
        "fits": "Code",
        "why": "An index lookup is deterministic and free; no model should do it.",
    },
    {
        "step": "Escalation gate",
        "fits": "Code",
        "why": "Thresholds over typed answers are plain code you can test and audit.",
    },
    {
        "step": "Redaction rewrite",
        "fits": "LLM",
        "why": "Writing the page with spans replaced is generation; Decisions only "
        "answers questions.",
    },
    {
        "step": "Privilege-log drafting",
        "fits": "LLM",
        "why": "A log entry is free text with an explanation; Decisions cannot "
        "write one.",
    },
    {
        "step": "QA tags",
        "fits": "LLM",
        "why": "Tags and notes are generated text; batch them off the page path.",
    },
]
WHEN_TO_USE = [
    {
        "factor": "Output",
        "use_decisions_when": "The answer is a typed route: yes/no, choice or score.",
        "use_an_llm_when": "The output is generated text or your own extraction "
        "schema.",
    },
    {
        "factor": "Volume",
        "use_decisions_when": "Millions of pages and the alternative is a large LLM "
        "(live, 6 pages: 4-6x cheaper than gpt-6.1-terra; about 1.7x pricier than "
        "gpt-6-luna as an LLM).",
        "use_an_llm_when": "Volume is low enough that per-call cost does not matter, "
        "or the smallest LLM is accurate enough.",
    },
    {
        "factor": "Latency",
        "use_decisions_when": "The page path is on a deadline (live, 6 pages: ~6x "
        "faster than gpt-6-luna and 6-8x faster than gpt-6.1-terra at p50).",
        "use_an_llm_when": "A second or two per page is acceptable.",
    },
    {
        "factor": "Tools",
        "use_decisions_when": "No tool calls are needed; code does the lookups.",
        "use_an_llm_when": "The model has to choose and call tools.",
    },
    {
        "factor": "Explanations",
        "use_decisions_when": "A probability is enough for routing.",
        "use_an_llm_when": "A reviewer needs a reason, such as a privilege-log entry.",
    },
    {
        "factor": "Model choice",
        "use_decisions_when": "gpt-6-luna is accurate enough for the rubric.",
        "use_an_llm_when": "You need a stronger model; Decisions only supports "
        "gpt-6-luna.",
    },
    {
        "factor": "Refusals",
        "use_decisions_when": "You treat a refused answer as escalate-to-attorney.",
        "use_an_llm_when": "You can retry or rephrase on a refusal.",
    },
    {
        "factor": "Confidence",
        "use_decisions_when": "You recalibrate thresholds on your own labelled split.",
        "use_an_llm_when": "You cannot recalibrate: confidence does not port between "
        "engines.",
    },
]


@dataclass(frozen=True)
class Replayed:
    outcome: PageOutcome
    trace_cost_usd: float
    privileged_leak: bool
    pii_leak: bool
    quality: float


@dataclass(frozen=True)
class PolicyScore:
    escalation_rate: float
    privileged_leaks: int
    pii_leaks: int
    mean_quality: float
    blended_cost_per_page_usd: float


class NoFeasiblePolicyError(ValueError):
    pass


def _redaction_cost(page: ReviewPage) -> float:
    usage = TokenUsage(
        input=approx_tokens(REDACTION_PROMPT) + approx_tokens(page.text),
        output=approx_tokens(scrub_pii(page)),
    )
    return estimate_cost_usd(MODELS.redaction, usage)


def replay(
    page: ReviewPage, result: ClassifierResult, policy: RoutingPolicy
) -> Replayed:
    route = decide(result, policy).route
    escalated = route == "attorney_review"
    produced = route in ("produce", "redact")
    if escalated:
        produced = scripted_attorney_decisions([page])[page.doc_id] == "produce"
    redacted = produced and (
        route == "redact" or (escalated and flags_pii(result, policy))
    )
    text = None
    if produced:
        text = scrub_pii(page) if redacted else page.text
    outcome = PageOutcome(
        route=route,
        decision="produce" if produced else "withhold",
        escalated=escalated,
        produced_text=text,
        tools=result.tools,
        parse_failed=result.parse_failed,
    )
    quality = evaluate_trace_quality(page, outcome)
    cost = result.cost_usd + (_redaction_cost(page) if redacted else 0.0)
    cost += ATTORNEY_REVIEW_COST_USD_PER_PAGE if escalated else 0.0
    return Replayed(
        outcome, cost, quality.privileged_leak, quality.pii_leak, quality.quality_score
    )


def evaluate_policy(
    results: Mapping[str, ClassifierResult],
    pages: Sequence[ReviewPage],
    policy: RoutingPolicy,
) -> PolicyScore:
    replayed = [replay(p, results[p.doc_id], policy) for p in pages]
    count = len(replayed)
    return PolicyScore(
        escalation_rate=sum(r.outcome.escalated for r in replayed) / count,
        privileged_leaks=sum(r.privileged_leak for r in replayed),
        pii_leaks=sum(r.pii_leak for r in replayed),
        mean_quality=sum(r.quality for r in replayed) / count,
        blended_cost_per_page_usd=sum(r.trace_cost_usd for r in replayed) / count,
    )


def _distance_from_default(policy: RoutingPolicy) -> float:
    return sum(
        abs(float(getattr(policy, name)) - float(getattr(DEFAULT_POLICY, name)))
        for name in DEFAULT_POLICY.__dataclass_fields__
    )


def calibrate_policy(
    results: Mapping[str, ClassifierResult],
    pages: Sequence[ReviewPage],
    grid: Calibration = SETTINGS.calibration,
) -> RoutingPolicy:
    candidates = itertools.product(
        grid.privilege_escalation,
        grid.pii_redaction,
        grid.not_responsive,
        grid.responsive,
        grid.responsive_confidence,
    )
    scored: list[tuple[float, float, RoutingPolicy]] = []
    for values in candidates:
        policy = RoutingPolicy(*values)
        score = evaluate_policy(results, pages, policy)
        if score.privileged_leaks == 0 and score.pii_leaks == 0:
            scored.append(
                (
                    round(score.blended_cost_per_page_usd, 9),
                    _distance_from_default(policy),
                    policy,
                )
            )
    if not scored:
        raise NoFeasiblePolicyError(
            "no policy in the grid has zero privileged and PII leaks"
        )
    return min(scored, key=lambda item: item[:2])[2]


def policy_report(
    results: Mapping[str, ClassifierResult],
    splits: Mapping[str, Sequence[ReviewPage]],
    policies: Mapping[str, RoutingPolicy],
) -> pd.DataFrame:
    rows = []
    for (split, pages), (name, policy) in itertools.product(
        splits.items(), policies.items()
    ):
        score = evaluate_policy(results, pages, policy)
        rows.append({"policy": name, "split": split, **score.__dict__})
    return pd.DataFrame(rows)


def _comparison_spec(name: str) -> Any:
    return replace(VARIANTS[CONTROLS], variant=name, label=ENGINE_LABELS[name])


def _simulated_engine(name: str) -> Classifier:
    if name == "agent":
        return SimulatedAgentClassifier(VARIANTS[CONTROLS])
    return SimulatedTypedClassifier(name)


async def simulate_engine_comparison(
    pages: Sequence[ReviewPage] = EVAL_SET, *, policy: RoutingPolicy = DEFAULT_POLICY
) -> dict[str, ReviewRun]:
    runs = {}
    for name in ENGINE_LABELS:
        runs[name] = await review_pages(
            _comparison_spec(name),
            pages,
            _simulated_engine(name),
            SimulatedRedactor(),
            policy=policy,
            mode="simulated",
        )
    return runs


async def live_engine_comparison(
    pages: Sequence[ReviewPage],
    *,
    client: AsyncOpenAI,
    budget: BudgetTracker,
) -> dict[str, ReviewRun]:
    client = client.with_options(timeout=LIVE.timeout_s)
    classifiers: dict[str, Classifier] = {
        "decisions": DecisionsClassifier(client),
        "llm-luna": LLMProbabilityClassifier(client, MODELS.luna),
        "llm-terra": LLMProbabilityClassifier(client, MODELS.terra),
        "agent": LiveAgentClassifier(client, CONTROLS),
    }
    runs = {}
    for name, classifier in classifiers.items():
        try:
            runs[name] = await run_traced(
                name,
                _comparison_spec(name),
                list(pages)[:LIVE_PAGE_LIMIT],
                classifier,
                OpenAIRedactor(client),
                budget=budget,
                policy=DEFAULT_POLICY,
            )
        except BudgetExceededError as error:
            logger.warning("engine %s stopped by the budget: %s", name, error)
            break
    return runs


def _routes(traces: Sequence[Trace]) -> dict[str, str]:
    return {t.doc_id: t.route for t in traces}


def engine_comparison_table(runs: Mapping[str, ReviewRun]) -> pd.DataFrame:
    reference = _routes(runs["decisions"].traces)
    rows = []
    for name, run in runs.items():
        frame = traces_frame(run.traces)
        routes = _routes(run.traces)
        agree = sum(routes[d] == reference.get(d) for d in routes) / len(routes)
        cost = frame["model_cost_usd"].mean()
        rows.append(
            {
                "engine": name,
                "label": ENGINE_LABELS[name],
                "accuracy_vs_labels": frame["route_correct"].mean(),
                "agreement_vs_decisions": agree,
                "cost_per_page_usd": cost,
                "cost_per_100k_pages_usd": cost * MONTHLY_PAGES,
                "p50_latency_s": frame["latency_s"].median(),
                "mean_sync_tokens": frame["sync_tokens"].mean(),
                "escalation_rate": frame["escalated"].mean(),
                "privileged_leaks": int(frame["privileged_leak"].sum()),
                "pii_leaks": int(frame["pii_leak"].sum()),
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(
    runs: Mapping[str, ReviewRun],
    pages: Sequence[ReviewPage],
    thresholds: Sequence[float] = SETTINGS.calibration.sweep_thresholds,
) -> pd.DataFrame:
    rows = []
    for name in TYPED_ENGINES:
        results = {
            r.doc_id: r.classification
            for r in runs[name].outcome.results
            if r.classification is not None
        }
        for threshold in thresholds:
            policy = replace(DEFAULT_POLICY, privilege_escalation=threshold)
            score = evaluate_policy(results, pages, policy)
            rows.append(
                {
                    "engine": name,
                    "privilege_threshold": threshold,
                    "escalation_rate": score.escalation_rate,
                    "privileged_leaks": score.privileged_leaks,
                }
            )
    return pd.DataFrame(rows)


async def cascade_analysis(pages: Sequence[ReviewPage] = EVAL_SET) -> pd.DataFrame:
    arms = {
        "Decisions only": await simulate_variant(ROUTER_ONLY, pages),
        "Decisions + terra cascade": await simulate_variant(DECISION_MODEL, pages),
    }
    terra = await review_pages(
        _comparison_spec("llm-terra"),
        pages,
        SimulatedTypedClassifier("llm-terra"),
        SimulatedRedactor(),
        policy=CALIBRATED_POLICY,
        mode="simulated",
    )
    arms["Terra only"] = terra.traces
    rows = []
    for arm, traces in arms.items():
        frame = traces_frame(traces)
        rows.append(
            {
                "arm": arm,
                "model_cost_per_page_usd": frame["model_cost_usd"].mean(),
                "attorney_escalations": int(frame["escalated"].sum()),
                "attorney_cost_per_page_usd": frame["attorney_cost_usd"].mean(),
                "blended_cost_per_page_usd": frame["cost_usd"].mean(),
                "mean_quality": frame["quality_score"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_engine_tradeoff(table: pd.DataFrame) -> Figure:
    engines: list[str] = table["engine"].astype(str).tolist()
    costs: list[float] = table["cost_per_page_usd"].astype(float).tolist()
    latency: list[float] = table["p50_latency_s"].astype(float).tolist()
    fastest = engines[latency.index(min(latency))]
    cheapest = engines[costs.index(min(costs))]
    live_cheapest = SETTINGS.live_run.cheapest_engine
    verdict = (
        f"{ENGINE_SHORT[fastest]} is fastest and cheapest"
        if fastest == cheapest
        else f"{ENGINE_SHORT[fastest]} is fastest, {ENGINE_SHORT[cheapest]} is cheapest"
    )
    caveat = (
        f"; live {ENGINE_SHORT[live_cheapest]} was cheaper"
        if live_cheapest != cheapest
        else ""
    )
    title = f"Simulated: {verdict}{caveat}"
    fig, (axis,) = new_figure(title)
    for engine, cost, seconds in zip(engines, costs, latency, strict=True):
        axis.scatter(
            cost,
            seconds,
            s=CHART.marker_size**2 * 2,
            color=color(engine),
            label=ENGINE_SHORT[engine],
            zorder=3,
        )
        axis.annotate(
            ENGINE_SHORT[engine],
            (cost, seconds),
            xytext=(CHART.label_pad * 2, CHART.label_pad * 2),
            textcoords="offset points",
            fontsize=CHART.font_size,
            color=PALETTE.text,
        )
    axis.set_xscale("log")
    axis.margins(x=0.3, y=0.2)
    axis.set_ylim(bottom=0)
    axis.annotate(
        "better: cheaper and faster",
        xy=(0.02, 0.03),
        xytext=(0.2, 0.2),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=CHART.font_size,
        color=PALETTE.text,
        arrowprops={"arrowstyle": "->", "color": PALETTE.text},
    )
    style_axis(
        axis,
        xlabel="model cost per page (USD, log scale, lower is better)",
        ylabel="p50 latency (seconds, lower is better)",
        grid_axis="y",
    )
    add_legend(axis, "upper left")
    return fig


def plot_threshold_sweep(sweep: pd.DataFrame) -> Figure:
    leaks = sweep[sweep["privileged_leaks"] > 0]
    first_leak = leaks.groupby("engine", sort=False)["privilege_threshold"].min()
    clean = [
        ENGINE_SHORT[str(e)] for e in sweep["engine"].unique() if e not in first_leak
    ]
    leaking = [
        f"{ENGINE_SHORT[str(engine)]} leaks privileged pages from {threshold:.2f}"
        if index == 0
        else f"{ENGINE_SHORT[str(engine)]} from {threshold:.2f}"
        for index, (engine, threshold) in enumerate(first_leak.items())
    ]
    title = (
        f"Simulated: {', '.join(leaking)}"
        + (f"; {', '.join(clean)} never leaks" if clean else "")
        if leaking
        else "Simulated: no privileged page leaks at any threshold"
    )
    fig, (axis,) = new_figure(title)
    for engine, group in sweep.groupby("engine", sort=False):
        name = str(engine)
        ordered = group.sort_values("privilege_threshold")
        thresholds: list[float] = ordered["privilege_threshold"].astype(float).tolist()
        rates: list[float] = ordered["escalation_rate"].astype(float).tolist()
        axis.plot(
            thresholds,
            rates,
            marker="o",
            color=color(name),
            linewidth=CHART.line_width,
            markersize=CHART.marker_size,
            zorder=3,
        )
        axis.annotate(
            ENGINE_SHORT[name],
            (thresholds[-1], rates[-1]),
            xytext=(CHART.label_pad * 4, 0),
            textcoords="offset points",
            va="center",
            fontsize=CHART.font_size,
            color=PALETTE.text,
        )
    leak_points = zip(
        leaks["privilege_threshold"].astype(float),
        leaks["escalation_rate"].astype(float),
        strict=True,
    )
    for threshold, rate in leak_points:
        axis.plot(
            threshold,
            rate,
            marker="X",
            color=PALETTE.critical,
            markersize=CHART.marker_size * 1.6,
            linestyle="none",
            zorder=4,
        )
    axis.plot(
        [],
        [],
        marker="X",
        color=PALETTE.critical,
        linestyle="none",
        label="✗ a privileged page would be produced",
        markersize=CHART.marker_size,
    )
    axis.set_xlim(0, CHART.threshold_axis_max)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    style_axis(
        axis,
        xlabel="privilege escalation threshold (a higher threshold escalates less)",
        ylabel="pages sent to an attorney",
    )
    add_legend(axis)
    return fig


def plot_calibration(policy_table: pd.DataFrame) -> Figure:
    splits = list(dict.fromkeys(policy_table["split"]))
    leaks = int(
        policy_table["privileged_leaks"].sum() + policy_table["pii_leaks"].sum()
    )
    rate = {
        (str(policy), str(split)): float(value)
        for policy, split, value in zip(
            policy_table["policy"],
            policy_table["split"],
            policy_table["escalation_rate"].astype(float),
            strict=True,
        )
    }
    shift = rate["calibrated", splits[-1]] - rate["default", splits[-1]]
    fig, (axis,) = new_figure(
        f"Calibration moves {splits[-1]} escalation by {shift * 100:+.0f} points "
        f"with {leaks} leaks"
    )
    width = CHART.bar_width / 2
    for offset, policy in ((-width / 2, "default"), (width / 2, "calibrated")):
        positions = [i + offset for i in range(len(splits))]
        values = [rate[policy, split] for split in splits]
        axis.bar(
            positions,
            values,
            width=width,
            color=color(f"{policy}_policy"),
            label=f"{policy} thresholds",
            zorder=3,
        )
        label_bars(axis, positions, values, [f"{v:.0%}" for v in values])
    axis.set_xticks(range(len(splits)))
    axis.set_xticklabels(splits)
    axis.margins(y=0.4)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.annotate(
        f"{leaks} leaks (privileged or PII) in every run",
        (0.99, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=CHART.font_size,
        color=PALETTE.text,
        fontweight="bold",
    )
    style_axis(axis, ylabel="pages sent to an attorney")
    add_legend(axis, "upper left")
    return fig

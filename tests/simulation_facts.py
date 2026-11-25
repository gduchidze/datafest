"""Numbers quoted in notebook text, computed once from the simulation and the labels."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from doc_review_opt.agent.variants import CALIBRATED_POLICY, VARIANT_ORDER
from doc_review_opt.config import LiveEngine, LiveRound, LiveRun, get_settings
from doc_review_opt.experiment.metrics import (
    HoldoutReport,
    VariantMetrics,
    all_variant_metrics,
    classifications_by_page,
    holdout_check,
)
from doc_review_opt.experiment.simulation import simulate_all
from doc_review_opt.experiment.traces import Trace, traces_frame
from doc_review_opt.experiment.tradeoffs import (
    engine_comparison_table,
    simulate_engine_comparison,
)
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.reporting.reporting import summarize
from doc_review_opt.reporting.walkthrough import GateResult, gate


@dataclass(frozen=True)
class Facts:
    pages: int
    hard_pages: int
    floor: float
    traces: tuple[Trace, ...]
    summary: pd.DataFrame
    metrics: dict[str, VariantMetrics]
    gates: dict[str, GateResult]
    engines: pd.DataFrame
    holdout: HoldoutReport
    live_run: LiveRun

    def value(self, variant: str, column: str) -> float:
        row = self.summary.loc[self.summary["variant"] == variant, column]
        return float(row.iloc[0])

    def trace(self, variant: str, doc_id: str) -> Trace:
        return next(
            t for t in self.traces if t.variant == variant and t.doc_id == doc_id
        )

    def live(self, variant: str) -> LiveRound:
        return next(r for r in self.live_run.rounds if r.variant == variant)

    def live_fastest(self) -> LiveEngine:
        return min(self.live_run.engines, key=lambda e: e.p50_latency_s)

    def live_cheapest(self) -> LiveEngine:
        return min(self.live_run.engines, key=lambda e: e.model_cost_per_page_usd)

    def engine(self, name: str, column: str) -> float:
        row = self.engines.loc[self.engines["engine"] == name, column]
        return float(row.iloc[0])

    def half(self, policy: str, split: str) -> tuple[float, float, float]:
        score = next(
            s.score
            for s in self.holdout.scores
            if s.policy == policy and s.split == split
        )
        return (
            score.escalation_rate,
            score.mean_quality,
            score.blended_cost_per_page_usd,
        )


@lru_cache(maxsize=1)
def facts() -> Facts:
    traces = asyncio.run(simulate_all())
    runs = asyncio.run(simulate_engine_comparison(policy=CALIBRATED_POLICY))
    results = classifications_by_page(runs["decisions"])
    return Facts(
        pages=len(EVAL_SET),
        hard_pages=sum(p.labels.difficulty in ("trap", "borderline") for p in EVAL_SET),
        floor=get_settings().quality.guardrail_pass_floor,
        traces=tuple(traces),
        summary=summarize(traces_frame(traces)),
        metrics={m.variant: m for m in all_variant_metrics(traces, VARIANT_ORDER)},
        gates={v: gate(traces, v) for v in VARIANT_ORDER},
        engines=engine_comparison_table(runs),
        holdout=holdout_check(results),
        live_run=get_settings().live_run,
    )

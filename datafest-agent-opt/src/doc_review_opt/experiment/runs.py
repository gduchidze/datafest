"""Run one variant's pages through the review graph and turn the outcome into traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from doc_review_opt.agent.graph import (
    BatchOutcome,
    build_review_graph,
    review_checkpointer,
    run_batch,
)
from doc_review_opt.agent.protocols import Classifier, Redactor
from doc_review_opt.agent.routing import RoutingPolicy
from doc_review_opt.agent.variants import VariantSpec
from doc_review_opt.experiment.batch import background_work
from doc_review_opt.experiment.traces import (
    BackgroundWork,
    Mode,
    RunInfo,
    Trace,
    trace_from_result,
)
from doc_review_opt.models import Disposition, ReviewPage


@dataclass(frozen=True)
class ReviewRun:
    traces: list[Trace]
    outcome: BatchOutcome
    trace_url: str | None = None
    cost_usd: float = 0.0


def scripted_attorney_decisions(
    pages: Sequence[ReviewPage],
) -> dict[str, Disposition]:
    decisions: dict[str, Disposition] = {}
    for page in pages:
        labels = page.labels
        keep = (
            not labels.privileged
            and labels.expected_disposition != "withhold"
            and labels.responsive_level >= 1
        )
        decisions[page.doc_id] = "produce" if keep else "withhold"
    return decisions


def traces_from_outcome(
    spec: VariantSpec,
    pages: Sequence[ReviewPage],
    outcome: BatchOutcome,
    mode: Mode,
) -> list[Trace]:
    by_id = {result.doc_id: result for result in outcome.results}
    traces = []
    for page in pages:
        result = by_id[page.doc_id]
        background = BackgroundWork()
        if spec.config.async_background and mode != "live":
            background = background_work(page, escalated=result.escalated)
        run = RunInfo(spec.variant, spec.label, mode, background)
        traces.append(trace_from_result(page, result, run))
    return traces


async def review_pages(
    spec: VariantSpec,
    pages: Sequence[ReviewPage],
    classifier: Classifier,
    redactor: Redactor,
    *,
    policy: RoutingPolicy,
    mode: Mode,
    attorney_decisions: Mapping[str, Disposition] | None = None,
    max_concurrency: int | None = None,
) -> ReviewRun:
    graph = build_review_graph(
        classifier, redactor, review_checkpointer(), policy=policy
    )
    script = attorney_decisions or scripted_attorney_decisions(pages)
    outcome = await run_batch(
        graph,
        pages,
        thread_id=f"{mode}-{spec.variant}",
        attorney_decisions=script,
        max_concurrency=max_concurrency,
    )
    return ReviewRun(traces_from_outcome(spec, pages, outcome, mode), outcome)

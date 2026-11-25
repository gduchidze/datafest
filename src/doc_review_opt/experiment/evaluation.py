"""Deterministic guardrails plus an optional LLM judge over existing traces."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pandas as pd
from openai import AsyncOpenAI, OpenAIError

from doc_review_opt.exceptions import BudgetExceededError
from doc_review_opt.experiment.live import (
    JUDGE_MODEL,
    BudgetedJudge,
    BudgetTracker,
    JudgeGrade,
)
from doc_review_opt.experiment.traces import Trace, deterministic_guardrail_check
from doc_review_opt.models import ReviewPage

NOT_RUN_REASON = "Set RUN_LLM_JUDGE=true to enable the optional judge."


async def _grade(
    page: ReviewPage, trace: Trace, judge: BudgetedJudge | None
) -> JudgeGrade:
    if judge is None:
        return JudgeGrade(reason=NOT_RUN_REASON)
    try:
        return await judge.grade(page, trace)
    except BudgetExceededError:
        return JudgeGrade(
            judge_status="error", reason="Budget reached before this trace; no grade."
        )
    except OpenAIError as error:
        return JudgeGrade(
            judge_status="error",
            reason=f"Judge request failed ({type(error).__name__}); no grade.",
        )


async def evaluate_answer_traces(
    pages: Sequence[ReviewPage],
    traces: Sequence[Trace],
    *,
    client: AsyncOpenAI | None = None,
    budget: BudgetTracker | None = None,
) -> pd.DataFrame:
    if client is not None and budget is None:
        raise ValueError("a judge client needs a budget tracker")
    judge = None if client is None or budget is None else BudgetedJudge(client, budget)
    by_id = {page.doc_id: page for page in pages}
    grades = await asyncio.gather(*(_grade(by_id[t.doc_id], t, judge) for t in traces))
    rows = []
    for trace, grade in zip(traces, grades, strict=True):
        failures = deterministic_guardrail_check(by_id[trace.doc_id], trace)
        rows.append(
            {
                "variant": trace.variant,
                "variant_label": trace.variant_label,
                "doc_id": trace.doc_id,
                "deterministic_passed": not failures,
                "deterministic_failures": ", ".join(failures),
                "judge_model": JUDGE_MODEL,
                "judge_status": grade.judge_status,
                "passed": grade.passed,
                "reason": grade.reason,
                "judge_cost_usd": grade.judge_cost_usd,
            }
        )
    return pd.DataFrame(rows)


def summarize_answer_evals(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in results.groupby("variant", sort=False):
        graded = group[group["judge_status"] == "graded"]
        judge_pass = graded["passed"].astype(bool)
        both = judge_pass & graded["deterministic_passed"]
        rows.append(
            {
                "variant": variant,
                "variant_label": group["variant_label"].iloc[0],
                "pages": len(group),
                "deterministic_pass_rate": group["deterministic_passed"].mean(),
                "judge_graded": len(graded),
                "judge_coverage": len(graded) / len(group),
                "judge_errors": int((group["judge_status"] == "error").sum()),
                "judge_pass_rate": judge_pass.mean() if len(graded) else float("nan"),
                "both_pass_rate": both.mean() if len(graded) else float("nan"),
                "known_judge_cost_usd": group["judge_cost_usd"].sum(min_count=1),
            }
        )
    return pd.DataFrame(rows)

"""Background work as Batch API rows (privilege log, QA, tags); never submitted."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from doc_review_opt.agent.classifiers.decisions import page_block
from doc_review_opt.config import get_settings
from doc_review_opt.experiment.traces import BackgroundWork, Trace
from doc_review_opt.models import ReviewPage, TokenUsage
from doc_review_opt.pricing import BATCH_DISCOUNT, approx_tokens, estimate_cost_usd

SETTINGS = get_settings()
BATCH_MODEL = SETTINGS.models.batch
BATCH_ENDPOINT = "/v1/responses"
BATCH_MAX_OUTPUT_TOKENS = SETTINGS.batch.max_output_tokens
BACKGROUND_OUTPUT_TOKENS = SETTINGS.batch.background_output_tokens

TASK_INSTRUCTIONS = {
    "privilege_log": (
        "Draft one privilege-log entry for this page: author, recipients, date, "
        "privilege asserted and a neutral description. Never quote the page."
    ),
    "qa_sample": (
        "Write a one-line quality-control note: does the coding of this page look "
        "consistent with the review protocol?"
    ),
    "production_tags": (
        "Return the production tags for this page: Bates prefix, confidentiality "
        "designation and volume."
    ),
}
ESCALATED_ONLY = frozenset({"privilege_log"})


def _row(page: ReviewPage, task: str) -> dict[str, Any]:
    return {
        "custom_id": f"{page.doc_id}:{task}",
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": BATCH_MODEL,
            "instructions": TASK_INSTRUCTIONS[task],
            "input": page_block(page),
            "reasoning": {"effort": SETTINGS.live.reasoning_effort},
            "text": {"verbosity": "low"},
            "max_output_tokens": BATCH_MAX_OUTPUT_TOKENS,
        },
    }


def _tasks(escalated: bool) -> list[str]:
    return [t for t in TASK_INSTRUCTIONS if escalated or t not in ESCALATED_ONLY]


def page_rows(page: ReviewPage, *, escalated: bool) -> list[dict[str, Any]]:
    return [_row(page, task) for task in _tasks(escalated)]


def background_work(page: ReviewPage, *, escalated: bool) -> BackgroundWork:
    rows = page_rows(page, escalated=escalated)
    usage = TokenUsage(
        input=sum(
            approx_tokens(r["body"]["instructions"] + r["body"]["input"]) for r in rows
        ),
        output=BACKGROUND_OUTPUT_TOKENS * len(rows),
    )
    return BackgroundWork(
        tokens=usage.input + usage.output,
        cost_usd=estimate_cost_usd(BATCH_MODEL, usage, discount=BATCH_DISCOUNT),
    )


def build_background_batch(
    pages: Sequence[ReviewPage], traces: Sequence[Trace] | None = None
) -> list[dict[str, Any]]:
    escalated = {t.doc_id: t.escalated for t in traces} if traces else {}
    rows: list[dict[str, Any]] = []
    for page in pages:
        rows.extend(page_rows(page, escalated=escalated.get(page.doc_id, True)))
    return rows


def write_batch_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path

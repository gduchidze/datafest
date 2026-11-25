"""Tests for the round helpers used by the demo notebook."""

import asyncio

import pandas as pd
import pytest

from doc_review_opt.agent.graph import (
    build_review_graph,
    review_checkpointer,
    run_batch,
)
from doc_review_opt.agent.variants import (
    BASELINE,
    CONTROLS,
    DECISION_MODEL,
    ORIGINAL_WORDING_VARIANT,
    ROUTING,
    SPLIT,
    VARIANT_LABELS,
    VARIANT_ORDER,
    VARIANTS,
    default_policy,
)
from doc_review_opt.config import get_settings
from doc_review_opt.experiment.metrics import classifications_by_page, holdout_check
from doc_review_opt.experiment.simulation import build_simulated_engines, simulate_all
from doc_review_opt.experiment.traces import Trace, traces_frame
from doc_review_opt.experiment.tradeoffs import simulate_engine_comparison
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.reporting.reporting import plot_quality_gate, summarize
from doc_review_opt.reporting.walkthrough import (
    before_after,
    confusion_table,
    follow_page,
    gate,
    gate_table,
    holdout_table,
    metrics_table,
    outcome_table,
    page_table,
    request_diff,
    symptoms,
    watch_percent,
)
from tests.test_metrics import TRACES

FLOOR = get_settings().quality.guardrail_pass_floor
LOW_PAGE = next(
    p for p in EVAL_SET if p.labels.risk == "low" and not p.labels.privileged
)
ROUTES = ("produce", "withhold", "redact", "attorney_review")
DEMO_PAGES = [next(p for p in EVAL_SET if p.labels.expected_route == r) for r in ROUTES]
PAGE_ROWS = [
    [
        "ARC-000141",
        "Q3 2024 REVENUE SUMMARY (INTERNAL)",
        "Finance Reporting",
        "produce",
    ],
    ["ARC-000212", "FACILITIES NOTICE", "Ines Calloway", "withhold"],
    ["ARC-000352", "SALES OPS EMAIL", "Noor Haddad", "redact"],
    ["ARC-000284", "EMAIL", "Priya Raghunathan (attorney)", "attorney_review"],
]


@pytest.fixture(scope="module")
def traces() -> list[Trace]:
    return asyncio.run(simulate_all())


@pytest.fixture(scope="module")
def all_traces() -> list[Trace]:
    return asyncio.run(simulate_all(include_extras=True))


@pytest.fixture(scope="module")
def summary(traces: list[Trace]) -> pd.DataFrame:
    return summarize(traces_frame(traces))


def test_baseline_is_rejected_and_controls_are_accepted(traces: list[Trace]) -> None:
    baseline = gate(traces, BASELINE)
    controls = gate(traces, CONTROLS)
    assert not baseline.accepted
    assert baseline.pii_leaks > 0
    assert controls.accepted
    assert str(controls).startswith(f"{VARIANT_LABELS[CONTROLS]}: ACCEPTED (")
    assert f"floor {FLOOR:.0%}" in str(controls)
    assert str(controls).endswith("failures: none)")


def test_rejected_gate_lists_failures(traces: list[Trace]) -> None:
    text = str(gate(traces, BASELINE))
    assert "REJECTED" in text
    assert "pii_in_produced_text" in text


def test_original_wording_is_rejected_on_the_floor_alone(
    all_traces: list[Trace],
) -> None:
    result = gate(all_traces, ORIGINAL_WORDING_VARIANT)
    assert not result.accepted
    assert result.privileged_leaks == 0
    assert result.pii_leaks == 0
    assert result.pass_rate == pytest.approx(56 / 60)
    assert result.pass_rate < FLOOR


def test_routing_is_accepted_with_one_missed_escalation(traces: list[Trace]) -> None:
    result = gate(traces, ROUTING)
    assert result.accepted
    assert result.pass_rate == pytest.approx(59 / 60)
    assert result.failures == {"escalation_mismatch": 1}


def test_gate_table_feeds_the_quality_chart(traces: list[Trace]) -> None:
    table = gate_table(traces, VARIANT_ORDER)
    assert list(table.columns) == ["label", "pass_rate", "accepted"]
    assert list(table["label"]) == [VARIANT_LABELS[v] for v in VARIANT_ORDER]
    assert plot_quality_gate(table, FLOOR) is not None


def test_symptoms_show_the_published_baseline_numbers(summary: pd.DataFrame) -> None:
    text = symptoms(summary, BASELINE)
    for fragment in (
        "5.0 per page (4.2 unneeded)",
        "8,474",
        "3.59s",
        "$0.00895",
        "47%",
        "privileged 0, PII 2",
    ):
        assert fragment in text
    assert "guardrail failures" not in text


def test_request_diff_shows_only_changed_settings() -> None:
    diff = request_diff(BASELINE, CONTROLS, LOW_PAGE)
    assert list(diff.columns) == ["setting", "before", "after"]
    assert "structured_output" in set(diff["setting"])
    assert (diff["before"] != diff["after"]).all()


def test_request_diff_is_empty_when_nothing_differs() -> None:
    assert request_diff(CONTROLS, CONTROLS, LOW_PAGE).empty


def test_request_diff_shows_the_model_change_for_routing() -> None:
    diff = request_diff(CONTROLS, ROUTING, LOW_PAGE)
    assert "model" in set(diff["setting"])


def test_before_after_has_one_formatted_row_per_metric(summary: pd.DataFrame) -> None:
    table = before_after(summary, BASELINE, CONTROLS)
    assert list(table.columns) == ["metric", "before", "after"]
    assert len(table) == 9
    before = dict(zip(table["metric"].tolist(), table["before"].tolist(), strict=True))
    after = dict(zip(table["metric"].tolist(), table["after"].tolist(), strict=True))
    assert before["privileged leaks"] == "0"
    assert before["PII leaks"] == "2"
    assert after["PII leaks"] == "0"
    assert before["model $/page"] == "$0.00895"
    assert after["model $/page"] == "$0.00136"
    assert after["model $/page"].startswith("$0.")
    assert after["escalation rate"].endswith("%")


def test_page_table_shows_each_page_in_order() -> None:
    table = page_table(DEMO_PAGES)
    assert list(table.columns) == ["doc_id", "page", "from", "expected"]
    assert table.to_numpy().tolist() == PAGE_ROWS


def test_page_table_drops_only_the_leading_title_prefix() -> None:
    copy = next(p for p in EVAL_SET if p.doc_id == "ARC-000572")
    assert page_table([copy])["page"].tolist() == [
        "Q3 2024 REVENUE SUMMARY (INTERNAL) -- COPY"
    ]


async def test_page_table_adds_route_and_decision_after_a_resumed_run() -> None:
    policy = default_policy(VARIANTS[BASELINE])
    engine, redactor = build_simulated_engines(VARIANTS[BASELINE], EVAL_SET, policy)
    graph = build_review_graph(engine, redactor, review_checkpointer(), policy=policy)
    outcome = await run_batch(
        graph, DEMO_PAGES, thread_id="page-table", attorney_decisions={}
    )
    assert [p["doc_id"] for p in outcome.interrupts] == ["ARC-000284"]
    table = page_table(DEMO_PAGES, outcome.results[::-1])
    assert table[["doc_id", "page", "from", "expected"]].to_numpy().tolist() == (
        PAGE_ROWS
    )
    assert list(table["route"]) == list(ROUTES)
    assert list(table["decision"]) == ["produce", "withhold", "produce", "withhold"]


def cells(table: pd.DataFrame, metric: str) -> dict[str, str]:
    row = table[table["metric"] == metric].iloc[0]
    return {label: str(row[label]) for label in table.columns[2:]}


def test_metrics_table_puts_hard_gates_above_reported_metrics(
    traces: list[Trace],
) -> None:
    table = metrics_table(traces, VARIANT_ORDER)
    labels = [VARIANT_LABELS[v] for v in VARIANT_ORDER]
    assert list(table.columns) == ["kind", "metric", *labels]
    kinds = list(table["kind"])
    assert set(kinds) == {"hard gate", "reported"}
    assert kinds == sorted(kinds, key=lambda kind: kind != "hard gate")
    assert kinds.count("hard gate") == 4


def test_metrics_table_hard_gate_rows_match_the_gate(traces: list[Trace]) -> None:
    table = metrics_table(traces, VARIANT_ORDER)
    accepted = cells(table, "round accepted")
    assert accepted[VARIANT_LABELS[BASELINE]] == "no"
    assert accepted[VARIANT_LABELS[CONTROLS]] == "yes"
    pii = cells(table, "PII leaked pages (must be 0)")
    assert pii[VARIANT_LABELS[BASELINE]] == "2"
    assert pii[VARIANT_LABELS[CONTROLS]] == "0"
    passed = cells(table, "guardrail pass rate (floor 95%)")
    assert passed[VARIANT_LABELS[ROUTING]] == "98%"


def test_metrics_table_marks_watch_cells_only_where_a_floor_is_missed(
    traces: list[Trace],
) -> None:
    table = metrics_table(traces, VARIANT_ORDER)
    under = cells(table, "under-escalation rate")
    assert under[VARIANT_LABELS[ROUTING]] == "5%"
    assert not under[VARIANT_LABELS[DECISION_MODEL]].endswith("(watch)")
    pii = cells(table, "PII recall")
    assert pii[VARIANT_LABELS[BASELINE]] == "93% (watch)"
    assert pii[VARIANT_LABELS[CONTROLS]] == "100%"
    assert cells(table, "relevance recall")[VARIANT_LABELS[ROUTING]] == "97%"
    accuracy = cells(table, "route accuracy")
    assert accuracy[VARIANT_LABELS[BASELINE]] == "83% (watch)"


def test_metrics_table_shows_tail_latency_and_cost_with_attorney_time(
    traces: list[Trace],
) -> None:
    table = metrics_table(traces, [BASELINE])
    assert cells(table, "p95 latency (s)")[VARIANT_LABELS[BASELINE]] == "4.10"
    assert cells(table, "$/page with attorney time")[VARIANT_LABELS[BASELINE]] == (
        "$0.709"
    )


def test_confusion_table_is_expected_by_actual_for_the_decision_model(
    traces: list[Trace],
) -> None:
    table = confusion_table(traces, DECISION_MODEL)
    assert list(table.index) == list(ROUTES)
    assert list(table.columns) == list(ROUTES)
    assert table.index.name == "expected"
    assert table.columns.name == "actual"
    assert table.to_numpy().tolist() == [
        [18, 0, 0, 0],
        [0, 12, 0, 0],
        [0, 0, 10, 0],
        [0, 0, 0, 20],
    ]


def test_confusion_table_shows_the_baseline_errors(traces: list[Trace]) -> None:
    table = confusion_table(traces, BASELINE)
    assert int(table.to_numpy().sum()) == len(EVAL_SET)
    assert int(table.to_numpy().trace()) == 50


def test_holdout_table_has_one_row_per_policy_and_half() -> None:
    runs = asyncio.run(simulate_engine_comparison())
    report = holdout_check(classifications_by_page(runs["decisions"]), EVAL_SET)
    table = holdout_table(report)
    assert list(table.columns) == [
        "policy",
        "split",
        "escalation rate",
        "privileged leaks",
        "PII leaks",
        "quality",
        "$/page",
    ]
    assert len(table) == 4
    assert set(table["split"]) == {"calibration", "holdout"}
    assert set(table["policy"]) == {"default", "refit"}


def test_flagged_floor_cells_never_round_up_to_the_floor() -> None:
    assert watch_percent(0.996, 1.0, below=True) == "99%"
    assert watch_percent(0.946, 0.95, below=True) == "94%"
    assert watch_percent(0.767, 0.90, below=True) == "77%"
    assert watch_percent(0.857, 1.0, below=True) == "86%"


def test_flagged_ceiling_cells_never_round_down_to_the_ceiling() -> None:
    assert watch_percent(0.052, 0.05, below=False) == "6%"
    assert watch_percent(0.25, 0.20, below=False) == "25%"
    assert watch_percent(0.1, 0.05, below=False) == "10%"


def hand_traces() -> list[Trace]:
    first = [t for t in TRACES if t.doc_id == "P4"]
    second = [
        t.model_copy(
            update={
                "variant": "w",
                "variant_label": "Round W",
                "engine": "decisions",
                "models_used": ("gpt-6-luna", "gpt-6.1-terra"),
                "tool_calls": 0,
                "sync_tokens": 2400,
                "latency_s": 0.5,
                "model_cost_usd": 0.00042,
                "route_reason": "privilege_above_threshold",
                "attorney_decision": "withhold",
            }
        )
        for t in first
    ]
    first = [
        t.model_copy(
            update={
                "variant_label": "Round V",
                "tool_calls": 5,
                "tools": ("a", "b", "c", "d", "e"),
                "sync_tokens": 8200,
                "latency_s": 3.456,
                "model_cost_usd": 0.0123,
                "route_reason": "agent_disposition",
            }
        )
        for t in first
    ]
    return [*first, *second]


def test_follow_page_has_one_plain_row_per_round() -> None:
    table = follow_page(hand_traces(), "P4", ["v", "w"])
    assert list(table.columns) == [
        "round",
        "engine and model",
        "tools called",
        "tokens",
        "seconds",
        "model cost",
        "outcome",
        "reason",
    ]
    assert table["round"].tolist() == ["Round V", "Round W"]
    assert table["tools called"].tolist() == [5, 0]
    assert table["tokens"].tolist() == ["8,200", "2,400"]
    assert table["seconds"].tolist() == ["3.46", "0.50"]
    assert table["model cost"].tolist() == ["$0.01230", "$0.00042"]
    assert table["engine and model"].tolist() == [
        "stub: stub",
        "decisions: gpt-6-luna, gpt-6.1-terra",
    ]
    assert table["reason"].tolist() == [
        "agent disposition",
        "privilege above threshold",
    ]


def test_follow_page_names_the_attorney_outcome() -> None:
    table = follow_page(hand_traces(), "P4", ["v", "w"])
    assert table["outcome"].tolist() == [
        "attorney review",
        "attorney review, then withhold",
    ]


def test_follow_page_follows_the_requested_round_order() -> None:
    table = follow_page(hand_traces(), "P4", ["w", "v"])
    assert table["round"].tolist() == ["Round W", "Round V"]


def test_follow_page_rejects_a_page_missing_from_a_round() -> None:
    with pytest.raises(ValueError, match="no trace"):
        follow_page(hand_traces(), "P1", ["v", "w"])


def test_follow_page_on_the_real_traces_covers_every_round(
    traces: list[Trace],
) -> None:
    page = next(p for p in EVAL_SET if p.labels.privileged)
    table = follow_page(traces, page.doc_id)
    assert table["round"].tolist() == [VARIANT_LABELS[v] for v in VARIANT_ORDER]
    assert set(table["outcome"]) <= {
        "attorney review",
        "attorney review, then withhold",
    }
    assert table["tools called"].iloc[0] == 5


def test_follow_page_lists_a_repeated_model_once() -> None:
    twice = [
        t.model_copy(update={"models_used": ("gpt-6-luna", "gpt-6-luna")})
        for t in hand_traces()
        if t.variant == "v"
    ]
    table = follow_page(twice, "P4", ["v"])
    assert table["engine and model"].tolist() == ["stub: gpt-6-luna"]


def test_outcome_table_adds_the_round_route_and_reason_to_the_page_table(
    traces: list[Trace],
) -> None:
    ids = ("ARC-000615", "ARC-000643", "ARC-000649")
    pages = [p for p in EVAL_SET if p.doc_id in ids]
    table = outcome_table(traces, pages, SPLIT)
    assert list(table.columns) == [
        "doc_id",
        "page",
        "from",
        "expected",
        "route",
        "reason",
    ]
    by_id = {t.doc_id: t for t in traces if t.variant == SPLIT}
    assert table["doc_id"].tolist() == [p.doc_id for p in pages]
    assert table["route"].tolist() == [by_id[p.doc_id].route for p in pages]
    assert table["reason"].tolist() == [
        by_id[p.doc_id].route_reason.replace("_", " ") for p in pages
    ]


def test_outcome_table_rejects_a_page_without_a_trace(traces: list[Trace]) -> None:
    page = EVAL_SET[0]
    with pytest.raises(ValueError, match="no trace"):
        outcome_table([t for t in traces if t.doc_id != page.doc_id], [page], SPLIT)

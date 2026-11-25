"""Tests for the deterministic simulation of every round."""

import asyncio
import json

import pandas as pd
import pytest

from doc_review_opt.agent.classifiers.decisions import Wording
from doc_review_opt.agent.classifiers.tool_agent import live_config_for_page
from doc_review_opt.agent.graph import build_review_graph
from doc_review_opt.agent.routing import DEFAULT_POLICY
from doc_review_opt.agent.variants import (
    BASELINE,
    CACHING,
    CONTROLS,
    DECISION_MODEL,
    GIST_WORDING,
    ROUTER_ONLY,
    ROUTING,
    SPLIT,
    VARIANT_ORDER,
    VARIANTS,
)
from doc_review_opt.config import get_settings
from doc_review_opt.experiment.simulation import (
    SimulatedTypedClassifier,
    _reply_text,
    decisions_latency_s,
    jitter,
    responses_latency_s,
    simulate_all,
    simulate_variant,
    simulated_answers,
    variant_request_example,
)
from doc_review_opt.experiment.traces import Trace, traces_frame
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.models import PredicateAnswer, ReviewPage, ScoreAnswer, TokenUsage
from doc_review_opt.reporting.reporting import summarize
from tests.helpers import at, row

SETTINGS = get_settings()


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return traces_frame(asyncio.run(simulate_all()))


@pytest.fixture(scope="module")
def summary(frame: pd.DataFrame) -> pd.DataFrame:
    return summarize(frame).set_index("variant")


def test_all_six_variants_cover_every_page(frame: pd.DataFrame) -> None:
    assert list(frame["variant"].unique()) == VARIANT_ORDER
    assert (frame.groupby("variant")["doc_id"].count() == len(EVAL_SET)).all()
    assert set(frame["mode"]) == {"simulated"}


async def test_simulation_is_deterministic() -> None:
    first = await simulate_variant(DECISION_MODEL)
    second = await simulate_variant(DECISION_MODEL)
    assert first == second
    assert isinstance(first[0], Trace)


def test_jitter_is_hash_seeded_and_in_range() -> None:
    assert jitter("ARC-1", "decisions") == jitter("ARC-1", "decisions")
    assert jitter("ARC-1", "decisions") != jitter("ARC-2", "decisions")
    assert 0 <= jitter("ARC-1", "decisions", "x") < 1


async def test_simulation_runs_through_the_real_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    real = build_review_graph

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(args[0])
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("doc_review_opt.experiment.runs.build_review_graph", spy)
    traces = await simulate_variant(CONTROLS)
    assert len(calls) == 1
    assert {t.route_reason for t in traces} >= {"agent_disposition"}


def test_baseline_is_worse_than_round_one(summary: pd.DataFrame) -> None:
    baseline, controls = row(summary, BASELINE), row(summary, CONTROLS)
    assert baseline["mean_quality"] < controls["mean_quality"]
    assert baseline["model_cost_per_page_usd"] > 5 * controls["model_cost_per_page_usd"]
    assert baseline["mean_sync_tokens"] > 3 * controls["mean_sync_tokens"]
    assert baseline["pii_leaks"] >= 1


def test_routing_shows_the_luna_slip_honestly(summary: pd.DataFrame) -> None:
    routing = row(summary, ROUTING)
    assert routing["policy_compliance"] < 1.0
    assert routing["model_cost_per_page_usd"] < at(
        summary, CONTROLS, "model_cost_per_page_usd"
    )
    assert routing["privileged_leaks"] == 0


def test_hard_gates_hold_for_every_round_after_the_baseline(
    summary: pd.DataFrame,
) -> None:
    later = summary.drop(index=BASELINE)
    assert (later["privileged_leaks"] == 0).all()
    assert (later["pii_leaks"] == 0).all()
    assert (summary["privileged_leaks"] == 0).all()


def test_decisions_round_is_faster_than_the_agent(summary: pd.DataFrame) -> None:
    assert at(summary, DECISION_MODEL, "p50_latency_s") < at(
        summary, CONTROLS, "p50_latency_s"
    )


def test_caching_round_shows_cached_tokens_and_lower_cost(
    summary: pd.DataFrame,
) -> None:
    before, after = row(summary, DECISION_MODEL), row(summary, CACHING)
    assert before["mean_cached_tokens"] == 0
    assert after["mean_cached_tokens"] > 1000
    assert after["model_cost_per_page_usd"] < before["model_cost_per_page_usd"]


def test_cache_write_is_charged_once(frame: pd.DataFrame) -> None:
    rows = frame[(frame["variant"] == CACHING) & (frame["cache_write_tokens"] > 0)]
    assert len(rows) == 1
    assert rows.iloc[0]["doc_id"] == min(p.doc_id for p in EVAL_SET)


def test_split_round_moves_work_to_background(summary: pd.DataFrame) -> None:
    before, after = row(summary, CACHING), row(summary, SPLIT)
    assert before["background_cost_per_page_usd"] == 0
    assert after["background_cost_per_page_usd"] > 0
    assert after["p50_latency_s"] < before["p50_latency_s"]
    assert after["mean_total_tokens"] > after["mean_sync_tokens"]


async def test_default_policy_over_escalates_and_calibrated_does_not() -> None:
    uncalibrated = await simulate_variant(ROUTER_ONLY, policy=DEFAULT_POLICY)
    calibrated = await simulate_variant(ROUTER_ONLY)
    escalated = sum(t.escalated for t in uncalibrated)
    assert escalated > sum(t.escalated for t in calibrated) + 5
    assert any(t.route_reason == "low_responsive_confidence" for t in uncalibrated)
    assert not any(t.privileged_leak or t.pii_leak for t in calibrated)


def model_cost(traces: list[Trace]) -> float:
    return sum(t.model_cost_usd for t in traces)


async def test_default_policy_makes_the_cascade_pay_for_second_opinions() -> None:
    uncalibrated = await simulate_variant(DECISION_MODEL, policy=DEFAULT_POLICY)
    calibrated = await simulate_variant(DECISION_MODEL)
    assert model_cost(uncalibrated) > 1.5 * model_cost(calibrated)
    assert not any(t.privileged_leak or t.pii_leak for t in calibrated)


async def test_simulated_second_opinion_prices_the_chosen_wording() -> None:
    page = EVAL_SET[0]
    gist = await SimulatedTypedClassifier("llm-terra", wording="gist").classify(
        page, "protocol"
    )
    protocol = await SimulatedTypedClassifier("llm-terra").classify(page, "protocol")
    assert gist.usage.input < protocol.usage.input


async def test_decisions_confidence_does_not_port() -> None:
    classifier = SimulatedTypedClassifier("decisions")
    decisive = [
        p
        for p in EVAL_SET
        if p.labels.difficulty == "simple" and p.labels.responsive_level == 2
    ]
    results = [await classifier.classify(p, "protocol") for p in decisive]
    scores = [r.answers["responsive"] for r in results]
    assert all(
        isinstance(s, ScoreAnswer) and 0.6 <= s.confidence <= 0.68 for s in scores
    )
    cc_business = next(p for p in EVAL_SET if p.cc and not p.labels.privileged)
    gist = SimulatedTypedClassifier("decisions", wording="gist")
    answers = (await gist.classify(cc_business, "protocol")).answers
    assert 0.2 < answers["privileged"].probability < 0.4  # type: ignore[union-attr]


async def test_prior_production_gate_catches_the_near_duplicate() -> None:
    traces = {t.doc_id: t for t in await simulate_variant(DECISION_MODEL)}
    duplicate = traces["ARC-000572"]
    assert (duplicate.engine, duplicate.decision, duplicate.model_cost_usd) == (
        "code",
        "withhold",
        0,
    )
    assert traces["ARC-000580"].escalated


def test_latency_formulas_are_labelled_and_monotonic() -> None:
    small = TokenUsage(input=1000, output=100)
    cached = TokenUsage(input=1000, cached=800, output=100)
    assert responses_latency_s("gpt-6.1-terra", cached) < responses_latency_s(
        "gpt-6.1-terra", small
    )
    assert responses_latency_s("gpt-6-luna", small) < responses_latency_s(
        "gpt-6.1-terra", small
    )
    assert decisions_latency_s(small) < responses_latency_s("gpt-6-luna", small)


def test_request_examples_differ_by_round() -> None:
    page = EVAL_SET[0]
    baseline = variant_request_example(BASELINE, page)
    controls = variant_request_example(CONTROLS, page)
    decisions = variant_request_example(CACHING, page)
    assert baseline["tool_choice"] == "required"
    assert controls["text"]["format"]["name"] == "review_disposition"
    assert baseline["reasoning"] == controls["reasoning"] == {"effort": "none"}
    assert "questions" in decisions
    assert "prompt_cache_key" in decisions


def privileged_probability(page: ReviewPage, wording: Wording) -> float:
    answer = simulated_answers(page, "decisions", wording)["privileged"]
    assert isinstance(answer, PredicateAnswer)
    return answer.probability


def test_wording_lowers_only_the_attorney_cc_business_pages() -> None:
    cc_pages = [p for p in EVAL_SET if p.cc and not p.labels.privileged]
    assert cc_pages
    for page in EVAL_SET:
        gist = privileged_probability(page, "gist")
        protocol = privileged_probability(page, "protocol")
        if page in cc_pages:
            limit = SETTINGS.policy.calibrated.privilege_escalation
            assert gist > limit > protocol
        else:
            assert gist == protocol


async def test_gist_wording_ablation_escalates_more_and_never_leaks() -> None:
    gist = await simulate_variant(GIST_WORDING)
    protocol = await simulate_variant(DECISION_MODEL)
    assert sum(t.escalated for t in gist) > sum(t.escalated for t in protocol)
    assert not any(t.privileged_leak for t in gist + protocol)


@pytest.mark.parametrize("variant", [BASELINE, CONTROLS, ROUTING])
def test_simulation_and_live_agree_on_strict_json_per_variant(variant: str) -> None:
    spec = VARIANTS[variant]
    live = live_config_for_page(EVAL_SET[0], variant)
    reply = _reply_text(spec, EVAL_SET[0], "produce", True)
    try:
        json.loads(reply)
        replied_in_json = True
    except json.JSONDecodeError:
        replied_in_json = False
    assert live.strict_json == spec.config.strict_json == replied_in_json
    assert spec.config.strict_json == (variant != BASELINE)

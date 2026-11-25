"""Pins numbers and claims quoted in the README and notebook text to computed values."""

from __future__ import annotations

import asyncio
import inspect
import itertools
import re
from collections import Counter
from pathlib import Path

import pytest

from doc_review_opt.agent.routing import decide
from doc_review_opt.agent.variants import (
    BASELINE,
    CACHING,
    CONTROLS,
    DECISION_MODEL,
    ORIGINAL_WORDING_VARIANT,
    ROUTING,
    SPLIT,
    VARIANT_ORDER,
)
from doc_review_opt.config import LiveRun
from doc_review_opt.experiment.live import pick_live_pages
from doc_review_opt.experiment.simulation import simulate_all
from doc_review_opt.experiment.tradeoffs import (
    ENGINE_SHORT,
    cascade_analysis,
    simulate_engine_comparison,
    threshold_sweep,
)
from doc_review_opt.fixtures import CALIBRATION, EVAL_SET, HOLDOUT
from doc_review_opt.reporting.walkthrough import gate, outcome_table
from tests.notebook_files import read_demo
from tests.simulation_facts import Facts, facts
from tests.test_eval_set import LIVE_IDS, ORIGINAL_IDS

README = (Path(__file__).resolve().parent.parent / "README.md").read_text("utf-8")
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
LIVE_PAGES = 6
TOLERANCE = 0.05


@pytest.fixture(scope="module")
def fx() -> Facts:
    return facts()


def readme_line(prefix: str) -> str:
    return next(line for line in README.splitlines() if line.startswith(prefix))


def test_readme_page_counts_and_split_match_the_eval_set() -> None:
    pages = len(EVAL_SET)
    line = readme_line("- **Pages.**")
    assert f"{pages} synthetic pages" in line
    assert f"plus {pages - LIVE_PAGES} new ones" in line
    split = readme_line("- **Split.**")
    assert f"calibration half of {len(CALIBRATION)} pages" in split
    assert f"holdout of {len(HOLDOUT)}" in split


def test_readme_route_and_difficulty_mix_match_the_labels() -> None:
    routes = Counter(p.labels.expected_route for p in EVAL_SET)
    expected = (
        f"{routes['produce']} produce, {routes['withhold']} withhold, "
        f"{routes['attorney_review']} attorney review, {routes['redact']} redact."
    )
    assert expected in readme_line("- **Expected routes.**")
    levels = Counter(p.labels.difficulty for p in EVAL_SET)
    expected = (
        f"{levels['simple']} simple, {levels['trap']} traps, "
        f"{levels['borderline']} borderline, {levels['routine']} routine."
    )
    assert expected in readme_line("- **Difficulty.**")


def test_readme_before_after_table_matches_the_simulation(fx: Facts) -> None:
    rows = {
        cells[0]: cells
        for cells in (
            [c.strip() for c in line.strip("|").split("|")]
            for line in README.splitlines()
            if line.startswith("| Bad baseline") or line.startswith("| Round ")
        )
        if len(cells) == 9
    }
    for _, row in fx.summary.iterrows():
        cells = rows[row.variant_label]
        assert cells[1:] == [
            f"{row.mean_quality:.2f}",
            str(int(row.privileged_leaks)),
            str(int(row.pii_leaks)),
            f"{row.escalation_rate:.0%}",
            f"{row.mean_sync_tokens:,.0f}",
            f"{row.p50_latency_s:.2f}s",
            f"${row.model_cost_per_page_usd:.5f}",
            f"${row.cost_per_page_usd:.3f}",
        ]


def test_readme_wording_ablation_numbers_match_the_simulation() -> None:
    traces = asyncio.run(simulate_all(include_extras=True))
    cc_pages = sum(1 for p in EVAL_SET if p.cc and not p.labels.privileged)
    assert f"{NUMBER_WORDS[cc_pages]} attorney-cc'd business emails" in README
    result = gate(traces, ORIGINAL_WORDING_VARIANT)
    assert f"rejected at {result.pass_rate:.0%} guardrail pass" in README


def test_readme_cost_ratio_matches_the_engine_comparison(fx: Facts) -> None:
    ratio = fx.engine("llm-terra", "cost_per_page_usd") / fx.engine(
        "decisions", "cost_per_page_usd"
    )
    match = re.search(r"costs about (\d+)x the Decisions call", README)
    assert match is not None
    assert int(match.group(1)) == round(ratio)


LIVE_CLAIM = "6 pages chosen from the original 30-page set, one per route in turn"


def test_the_live_run_pick_claim_matches_the_pick_function() -> None:
    picked = [p.doc_id for p in pick_live_pages(EVAL_SET)]
    assert tuple(picked) == LIVE_IDS
    assert len(picked) == LIVE_PAGES
    assert set(picked) < set(ORIGINAL_IDS)
    assert LIVE_CLAIM in README
    assert "first 6 pages" not in README


def test_demo_text_states_the_computed_page_counts(fx: Facts) -> None:
    text = "\n".join(c.source for c in read_demo().cells if c.cell_type == "markdown")
    assert f"{fx.pages} synthetic, labelled pages" in text
    assert f"{fx.hard_pages} are traps or borderline" in text
    assert LIVE_CLAIM in text
    assert "first 6 pages" not in text
    assert f"simulation above now uses all {fx.pages}" in text
    assert "all 30" not in text


def test_demo_baseline_symptoms_match_the_traces(fx: Facts) -> None:
    text = "\n".join(c.source for c in read_demo().cells if c.cell_type == "markdown")
    metrics = fx.metrics[BASELINE]
    extra = fx.value(BASELINE, "mean_extra_tool_calls")
    tokens = fx.value(BASELINE, "mean_sync_tokens") / 1000
    assert f"{extra:.1f} of 5 tool calls per page are unneeded" in text
    assert f"about {tokens:.1f}k tokens" in text
    assert (
        f"personal data left in produced text on {metrics.pii_leaked_pages} pages"
        in text
    )
    assert f"route accuracy {metrics.route_accuracy:.0%}" in text


def test_the_notebook_round_claims_hold(fx: Facts) -> None:
    cost = "model_cost_per_page_usd"
    latency = "p50_latency_s"
    rounds = [v for v in VARIANT_ORDER if v != BASELINE]
    assert fx.value(ROUTING, cost) == min(fx.value(v, cost) for v in rounds)
    assert fx.value(ROUTING, cost) < fx.value(BASELINE, cost)
    assert fx.gates[ROUTING].failures == {"escalation_mismatch": 1}
    assert fx.gates[DECISION_MODEL].failures == {}
    assert fx.value(DECISION_MODEL, latency) < fx.value(ROUTING, latency)
    assert fx.value(DECISION_MODEL, cost) > fx.value(ROUTING, cost)
    assert fx.value(CACHING, cost) < fx.value(DECISION_MODEL, cost)
    assert fx.value(SPLIT, latency) < fx.value(CACHING, latency)
    assert fx.value(SPLIT, cost) == pytest.approx(
        fx.value(CACHING, cost), rel=TOLERANCE
    )
    drops = [
        fx.value(a, cost) - fx.value(b, cost)
        for a, b in itertools.pairwise(VARIANT_ORDER)
    ]
    assert drops[0] == max(drops)
    assert fx.metrics[BASELINE].pii_leaked_pages > 0
    assert fx.metrics[CONTROLS].pii_leaked_pages == 0


def test_followed_pages_keep_their_outcome_in_every_round(fx: Facts) -> None:
    for variant in VARIANT_ORDER:
        privileged = fx.trace(variant, "ARC-000284")
        clean = fx.trace(variant, "ARC-000289")
        assert privileged.route == "attorney_review"
        assert clean.route == "produce"
    assert fx.trace(SPLIT, "ARC-000289").tool_calls == 0


def test_engine_comparison_claims_hold(fx: Facts) -> None:
    engines = fx.engines.set_index("engine")
    assert engines["cost_per_page_usd"].idxmin() == "decisions"
    assert engines["p50_latency_s"].idxmin() == "decisions"
    assert fx.engine("decisions", "escalation_rate") > fx.engine(
        "llm-terra", "escalation_rate"
    )


def test_probability_dial_claim_holds() -> None:
    sweep = threshold_sweep(asyncio.run(simulate_engine_comparison()), EVAL_SET)
    leaks = sweep.groupby("engine")["privileged_leaks"].max()
    assert leaks["decisions"] == 0
    assert leaks["llm-luna"] > 0
    assert leaks["llm-terra"] > 0


def test_holdout_claims_hold(fx: Facts) -> None:
    assert fx.holdout.holds
    refit = fx.half("refit", "holdout")
    default = fx.half("default", "holdout")
    assert refit[2] < default[2]
    assert refit[0] < default[0]


LATENCY_MARGIN = 3
PRICE_MARGIN = 5


def test_long_notebook_tradeoff_claims_hold() -> None:
    arms = asyncio.run(cascade_analysis()).set_index("arm")
    only, cascade, terra = (
        arms.loc["Decisions only"],
        arms.loc["Decisions + terra cascade"],
        arms.loc["Terra only"],
    )
    cost = "model_cost_per_page_usd"
    assert only[cost] < cascade[cost] < terra[cost]
    assert cascade["attorney_escalations"] < only["attorney_escalations"]
    assert cascade["blended_cost_per_page_usd"] < only["blended_cost_per_page_usd"]
    assert terra[cost] > PRICE_MARGIN * only[cost]
    assert terra["mean_quality"] <= cascade["mean_quality"]


def test_decisions_is_several_times_faster_and_far_cheaper_than_terra(
    fx: Facts,
) -> None:
    for llm in ("llm-luna", "llm-terra"):
        assert fx.engine("decisions", "p50_latency_s") * LATENCY_MARGIN < fx.engine(
            llm, "p50_latency_s"
        )
    assert fx.engine("decisions", "cost_per_page_usd") * PRICE_MARGIN < fx.engine(
        "llm-terra", "cost_per_page_usd"
    )


def test_no_live_run_page_is_attorney_cc_and_the_text_says_so() -> None:
    by_id = {p.doc_id: p for p in EVAL_SET}
    assert not any(by_id[i].cc for i in LIVE_IDS)
    assert "none of the 6 pages in the live run was attorney-cc'd" in README
    assert "None of the 6 pages above was attorney-cc'd" not in README
    assert "Separate live probe (none of the 6 pages above was attorney-cc'd)" in README


def test_demo_reader_text_is_plain(fx: Facts) -> None:
    text = "\n".join(c.source for c in read_demo().cells if c.cell_type == "markdown")
    assert "config.toml" not in text
    assert "p50 latency** is the typical page" in text
    assert "(their thresholds tuned on part of the pages)" in text
    assert "Decisions API (typed answers, not free text)" in text
    floor = (
        f"a pass rate of at least {fx.floor:.0%} of pages (the guardrail pass floor)"
    )
    assert floor in text
    assert text.index("p50 latency**") < text.index("p50 latency falls")


def demo_text() -> str:
    return "\n".join(c.source for c in read_demo().cells if c.cell_type == "markdown")


def test_round_two_text_matches_the_gate(fx: Facts) -> None:
    result = fx.gates[ROUTING]
    assert result.accepted
    assert result.pass_rate > fx.floor
    assert result.failures == {"escalation_mismatch": 1}
    text = demo_text()
    sentence = (
        f"1 miss in {fx.pages} pages stays above the {fx.floor:.0%} floor, "
        "so Round 2 is accepted"
    )
    assert sentence in text
    assert "The gate caught it" not in text
    assert "Every round must pass the quality gate" not in text
    assert "Cheapest is not best.** Check what each round gives up." in text


def failed_live_claims(run: LiveRun, sim_accuracy: float) -> set[str]:
    rounds = {r.variant: r for r in run.rounds}
    engines = {e.engine: e for e in run.engines}
    routing, controls = rounds[ROUTING], rounds[CONTROLS]
    decision, caching = rounds[DECISION_MODEL], rounds[CACHING]
    checks = {
        "round2_slower": routing.p50_latency_s > controls.p50_latency_s,
        "round2_lower_quality": routing.quality < controls.quality,
        "decisions_same_quality": decision.quality == controls.quality,
        "caching_costlier": caching.model_cost_per_page_usd
        > decision.model_cost_per_page_usd,
        "sim_below_perfect": sim_accuracy < 1,
        "decisions_fastest": min(engines.values(), key=lambda e: e.p50_latency_s).engine
        == "decisions",
        "luna_cheapest": min(
            engines.values(), key=lambda e: e.model_cost_per_page_usd
        ).engine
        == run.cheapest_engine
        == "llm-luna",
        "decisions_pricier_than_luna": engines["decisions"].model_cost_per_page_usd
        > engines["llm-luna"].model_cost_per_page_usd,
    }
    return {name for name, held in checks.items() if not held}


def perturbed(run: LiveRun, variant: str, **fields: float) -> LiveRun:
    rounds = tuple(
        r.model_copy(update=fields) if r.variant == variant else r for r in run.rounds
    )
    return run.model_copy(update={"rounds": rounds})


def perturbed_engine(run: LiveRun, engine: str, **fields: float) -> LiveRun:
    engines = tuple(
        e.model_copy(update=fields) if e.engine == engine else e for e in run.engines
    )
    return run.model_copy(update={"engines": engines})


def test_recorded_live_claims_hold_and_each_check_can_fail(fx: Facts) -> None:
    run = fx.live_run
    accuracy = fx.engine("decisions", "accuracy_vs_labels")
    assert failed_live_claims(run, accuracy) == set()
    cases = {
        "round2_slower": perturbed(run, ROUTING, p50_latency_s=0.5),
        "round2_lower_quality": perturbed(run, ROUTING, quality=0.98),
        "decisions_same_quality": perturbed(run, DECISION_MODEL, quality=0.9),
        "caching_costlier": perturbed(run, CACHING, model_cost_per_page_usd=0.00001),
        "decisions_fastest": perturbed_engine(run, "llm-luna", p50_latency_s=0.1),
        "luna_cheapest": perturbed_engine(
            run, "agent", model_cost_per_page_usd=0.00001
        ),
        "decisions_pricier_than_luna": perturbed_engine(
            run, "decisions", model_cost_per_page_usd=0.00005
        ),
    }
    for claim, broken in cases.items():
        assert claim in failed_live_claims(broken, accuracy)
    assert "sim_below_perfect" in failed_live_claims(run, 1.0)


def test_live_reading_text_quotes_the_recorded_figures(fx: Facts) -> None:
    routing, controls = fx.live(ROUTING), fx.live(CONTROLS)
    decision, caching = fx.live(DECISION_MODEL), fx.live(CACHING)
    fastest, cheapest = fx.live_fastest(), fx.live_cheapest()
    text = demo_text()
    for figure in (
        f"({routing.p50_latency_s:.2f}s vs {controls.p50_latency_s:.2f}s)",
        f"(quality {routing.quality:.2f} vs {controls.quality:.2f})",
        f"scored the same as Round 1 ({decision.quality:.2f}) on its "
        f"{fx.live_run.pages} pages",
        f"{fx.engine('decisions', 'accuracy_vs_labels'):.0%} route accuracy",
        f"(${caching.model_cost_per_page_usd:.5f} vs "
        f"${decision.model_cost_per_page_usd:.5f} per page)",
        f"{ENGINE_SHORT[fastest.engine]} was the fastest engine "
        f"({fastest.p50_latency_s:.2f}s)",
        f"{ENGINE_SHORT[cheapest.engine]} the cheapest "
        f"(${cheapest.model_cost_per_page_usd:.5f} per page)",
        "in the live rounds, routing was slower and caching cost more",
    ):
        assert figure in text
    assert "misroute" not in text.lower()
    assert "misroute" not in README.lower()
    assert "the fastest engine and `gpt-6-luna` as an LLM the cheapest" in README


TRICKY_IDS = (
    "ARC-000615",
    "ARC-000622",
    "ARC-000643",
    "ARC-000649",
    "ARC-000684",
    "ARC-000665",
)


def test_harder_pages_table_shows_the_round_five_routes(fx: Facts) -> None:
    pages = [p for i in TRICKY_IDS for p in EVAL_SET if p.doc_id == i]
    table = outcome_table(list(fx.traces), pages, SPLIT)
    for _, row in table.iterrows():
        trace = fx.trace(SPLIT, row["doc_id"])
        assert row["route"] == trace.route
        assert row["route"] == row["expected"]
    sources = [c.source for c in read_demo().cells if c.cell_type == "code"]
    assert any("outcome_table(traces, tricky_pages, SPLIT)" in s for s in sources)
    assert "what Round 5 did" in demo_text()


def test_the_printed_router_source_backs_the_fail_safe_claim() -> None:
    printed = inspect.getsource(decide)
    assert "missing_or_refused_answer" in printed
    assert "classifier_error" in printed
    sources = [c.source for c in read_demo().cells if c.cell_type == "code"]
    assert any("inspect.getsource(decide)" in s for s in sources)
    assert "never produces a page; it fails safe to an attorney" in demo_text()

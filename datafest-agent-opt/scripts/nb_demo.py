"""Cells for the condensed 20 minute demo notebook."""

# ruff: noqa: E501
from __future__ import annotations

from doc_review_opt.agent.variants import (
    BASELINE,
    CACHING,
    CONTROLS,
    DECISION_MODEL,
    ROUTING,
    SPLIT,
)
from doc_review_opt.experiment.tradeoffs import ENGINE_SHORT
from scripts.cells import Cell, code, md
from scripts.nb_facts import Facts, facts


def _round(header: str, diagram: str, previous: str, current: str) -> list[Cell]:
    return [
        md(header),
        code(f'draw_diagram("{diagram}")', hidden=True),
        code(
            f"""
display(before_after(summary, {previous}, {current}).style.hide(axis="index"))
print(gate(traces, {current}))
display(plot_progress(summary, upto_variant={current}))
""",
            hidden=True,
        ),
    ]


def _intro() -> list[Cell]:
    return [
        md("""
# Optimizing document review agents for cost and quality

**The promise:** find which steps need an LLM and which need only a decision model, and prove it with numbers.

A law firm's e-discovery agent reads each page of a company's files and decides what to hand to the other side in a lawsuit. We make it cheaper and faster, one change at a time.
"""),
        code(
            """
%matplotlib inline
import os

import pandas as pd
from dotenv import load_dotenv
from IPython.display import display
from openai import AsyncOpenAI

load_dotenv(override=False)
RUN_LIVE_API_CALLS = os.environ.get("RUN_LIVE_API_CALLS", "false").lower() == "true"
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
if RUN_LIVE_API_CALLS and not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY before enabling live API calls.")
if LANGSMITH_TRACING and not os.environ.get("LANGSMITH_API_KEY"):
    raise RuntimeError("Set LANGSMITH_API_KEY before enabling LangSmith tracing.")
os.environ["LANGSMITH_TRACING"] = str(LANGSMITH_TRACING).lower()
os.environ["LANGCHAIN_TRACING_V2"] = str(LANGSMITH_TRACING).lower()
os.environ.setdefault("LANGSMITH_PROJECT", "doc-review-opt")
from doc_review_opt.config import get_settings
from doc_review_opt.experiment.live import BudgetTracker
from doc_review_opt.reporting.diagrams import draw_diagram
LIVE = get_settings().live
client = AsyncOpenAI(timeout=LIVE.timeout_s, max_retries=LIVE.max_retries) if RUN_LIVE_API_CALLS else None
budget = BudgetTracker()
print(f"Live: {RUN_LIVE_API_CALLS} | LangSmith: {LANGSMITH_TRACING} | budget stop: ${budget.limit_usd:.2f}")
""",
            hidden=True,
        ),
        md("""
## Agenda: 20 minutes

| Steps | What | Min |
|---|---|---|
| 1-2 | The system and what we optimize | 3 |
| 3-4 | The test set and the starting point | 2 |
| 5-6 | One page, then harder pages | 2 |
| 7-11 | Five rounds, one change each | 8 |
| 12 | Results | 3 |
| 13-14 | Live check and takeaways | 2 |
"""),
    ]


def _system() -> list[Cell]:
    return [
        md("""
## Step 1: The system in one picture

A page goes in. One of four outcomes comes out: **produce**, **withhold**, **redact** (then produce) or **attorney review**.

Three parts decide:
1. **The engine** answers three questions: is the page relevant, is it privileged (lawyer advice), does it contain personal data?
2. **The router** is plain code. It turns the three answers into one outcome.
3. **An attorney** is asked only when the system is unsure or privilege is at stake.
"""),
        md("""
| The page is… | Outcome | Why it matters |
|---|---|---|
| Lawyer-client advice (privileged) | **attorney review**, then withhold | Producing it can waive privilege |
| Not relevant to the case | **withhold** | Nothing to hand over |
| Relevant, with personal data | **redact**, then produce | Personal data must not leak |
| Relevant and clean | **produce** | |
| Unclear | **attorney review** | Safe, but $1.50 of attorney time |
"""),
        md("""
The system runs as a LangGraph graph: boxes and arrows. `Send` starts one copy per page, in parallel. `classify` asks the engine. `attorney_review` calls `interrupt()`, which pauses the run until a person answers. Every round keeps this graph; only what runs inside the boxes changes.
"""),
        code(
            """
from doc_review_opt.agent.graph import build_review_graph, review_checkpointer
from doc_review_opt.agent.variants import BASELINE, VARIANTS, default_policy
from doc_review_opt.experiment.simulation import build_simulated_engines
from doc_review_opt.fixtures import EVAL_SET
from doc_review_opt.reporting.diagrams import draw_graph

policy = default_policy(VARIANTS[BASELINE])
engine, redactor = build_simulated_engines(VARIANTS[BASELINE], EVAL_SET, policy)
review_graph = build_review_graph(engine, redactor, review_checkpointer(), policy=policy)
draw_graph(review_graph.get_graph(xray=True))
""",
            hidden=True,
        ),
        md("""
Four pages, one per outcome. The privileged page stops at the attorney pause. The pause carries the page id and the reason, never page text. Then the attorney answers and the run finishes.
"""),
        code(
            """
from uuid import uuid4
from langgraph.types import Command
from doc_review_opt.reporting.walkthrough import page_table
OUTCOMES = ("produce", "withhold", "redact", "attorney_review")
demo_pages = [next(p for p in EVAL_SET if p.labels.expected_route == o) for o in OUTCOMES]
config = {"configurable": {"thread_id": f"intro-{uuid4()}"}}
paused = await review_graph.ainvoke({"pages": demo_pages, "results": []}, config)
waiting = [pending.value["doc_id"] for pending in paused["__interrupt__"]]
print(f"{len(paused['results'])} pages finished, {len(waiting)} waiting for an attorney: {', '.join(waiting)}")
resume = {pending.id: "withhold" for pending in paused["__interrupt__"]}
final = await review_graph.ainvoke(Command(resume=resume), config)
display(page_table(demo_pages, final["results"]).style.hide(axis="index"))
""",
            hidden=True,
        ),
        code('draw_diagram("human_in_the_loop")', hidden=True),
    ]


def _optimize() -> list[Cell]:
    fx = facts()
    return [
        md("""
## Step 2: What we optimize

**Make each page cheaper and faster without ever breaking the two hard rules:** never produce a privileged page, never leave personal data in a produced page.

| Lever | What it changes | Cost it attacks |
|---|---|---|
| Controls | Tight prompt, fewer tools, strict JSON | Tokens |
| Model routing | Cheap model for routine pages | Model price |
| Decision model | One call to the Decisions API (typed answers, not free text) plus a code router | Latency, price |
| Prompt caching | Reuse the unchanged prompt start | Input tokens |
| Workflow split | Side jobs move to Batch | Waiting time |

The same test runs after every change:
"""),
        code('draw_diagram("eval_pipeline")', hidden=True),
        md(f"""
A *trace* is the record of one page's run: what the engine did, tokens, seconds, cost, outcome. A *gate* is a pass/fail check. Hard gates and a pass rate of at least {fx.floor:.0%} of pages (the guardrail pass floor) accept or reject a round. Other metrics only warn, marked **(watch)** when below their floor. No LLM judge.
"""),
    ]


def _test_set() -> list[Cell]:
    fx = facts()
    return [
        md(f"""
## Step 3: The test set

{fx.pages} synthetic, labelled pages for one lawsuit; {fx.hard_pages} are traps or borderline. The first 30 were written first; 30 more add harder kinds: a lawyer only on cc, relayed advice, personal data in an irrelevant page, damaged scans, near-duplicates. Zero privileged pages produced and zero personal data left in produced text are the hard gates.
"""),
        code(
            """
from doc_review_opt.reporting.reporting import plot_eval_set

eval_table = pd.DataFrame(
    {
        "doc_id": page.doc_id,
        "expected_route": page.labels.expected_route,
        "difficulty": page.labels.difficulty,
    }
    for page in EVAL_SET
)
display(plot_eval_set(eval_table))
""",
            hidden=True,
        ),
    ]


def _baseline() -> list[Cell]:
    fx = facts()
    base = fx.metrics[BASELINE]
    return [
        md(f"""
## Step 4: The starting point

One `gpt-6.1-terra` agent does everything: every tool on every page, verbose payloads, free-text answer. We run it on all {fx.pages} pages (simulated, seconds). **p50 latency** is the typical page's time: half the pages are faster.
"""),
        code('draw_diagram("bad_baseline")', hidden=True),
        code(
            """
from doc_review_opt.agent.variants import BASELINE, CACHING, CALIBRATED_POLICY, CONTROLS, DECISION_MODEL, ROUTING, SPLIT, VARIANT_ORDER
from doc_review_opt.experiment.simulation import simulate_all
from doc_review_opt.experiment.traces import traces_frame
from doc_review_opt.reporting.reporting import plot_progress, plot_quality_gate, plot_round_chart, summarize
from doc_review_opt.reporting.walkthrough import before_after, confusion_table, follow_page, gate, gate_table, holdout_table, metrics_table, outcome_table, symptoms

traces = await simulate_all()
summary = summarize(traces_frame(traces))
SAMPLE_PAGE = next(p for p in EVAL_SET if p.labels.privileged)
CLEAN_PAGE = next(p for p in EVAL_SET if p.doc_id == "ARC-000289")
""",
            hidden=True,
        ),
        code(
            """
print(symptoms(summary, BASELINE))
print(gate(traces, BASELINE))
display(metrics_table(traces, [BASELINE]).style.hide(axis="index"))
""",
            hidden=True,
        ),
        md(f"""
**Errors found:** {fx.value(BASELINE, "mean_extra_tool_calls"):.1f} of {fx.value(BASELINE, "mean_tool_calls"):.0f} tool calls per page are unneeded, about {fx.value(BASELINE, "mean_sync_tokens") / 1000:.1f}k tokens per page, personal data left in produced text on {base.pii_leaked_pages} pages, route accuracy {base.route_accuracy:.0%}, {base.unnecessary_escalation_rate:.0%} needless attorney reviews. Each round fixes one thing.
"""),
    ]


def _follow_reading(fx: Facts) -> str:
    pv, pa = fx.trace(BASELINE, "ARC-000284"), fx.trace(SPLIT, "ARC-000284")
    cv, ca = fx.trace(BASELINE, "ARC-000289"), fx.trace(SPLIT, "ARC-000289")
    return (
        f"**The privileged page** goes to an attorney in every round; no lever changes the safe outcome. "
        f"Only the bill falls: {pv.sync_tokens:,} tokens and {pv.latency_s:.2f}s become {pa.sync_tokens:,} tokens and {pa.latency_s:.2f}s.\n\n"
        f"**The clean page** is produced in every round. Its cost falls from ${cv.model_cost_usd:.5f} to ${ca.model_cost_usd:.5f}, "
        f"and the {cv.tool_calls} tools the baseline called become {ca.tool_calls}."
    )


def _follow() -> list[Cell]:
    return [
        md("""
## Step 5: Follow one page through every round

Same page, six runs. Each row is one trace: engine, tools called, tokens, seconds, model cost, and the outcome with the router's reason.
"""),
        code("""
for page in (SAMPLE_PAGE, CLEAN_PAGE):
    display(page_table([page]).style.hide(axis="index"))
    display(follow_page(traces, page.doc_id).style.hide(axis="index"))
"""),
        md(_follow_reading(facts())),
    ]


def _examples() -> list[Cell]:
    return [
        md("""
## Step 6: Harder pages

Same rules, trickier pages. The table shows what the labels expect and what Round 5 did.

- **ARC-000615:** a lawyer is only on cc and nobody asks for advice.
- **ARC-000622:** a "privileged" stamp, but no lawyer and no advice.
- **ARC-000643:** full of personal data but irrelevant to the case.
- **ARC-000649:** a resent copy of pages already produced.
- **ARC-000684:** "our lawyers said", with no lawyer named.
- **ARC-000665:** a long workpaper with a phone number and email at the end.
"""),
        code(
            """
TRICKY = ("ARC-000615", "ARC-000622", "ARC-000643", "ARC-000649", "ARC-000684", "ARC-000665")
tricky_pages = [next(p for p in EVAL_SET if p.doc_id == doc_id) for doc_id in TRICKY]
display(outcome_table(traces, tricky_pages, SPLIT).style.hide(axis="index"))
""",
            hidden=True,
        ),
    ]


def _round_one() -> list[Cell]:
    fx = facts()
    before, after = fx.metrics[BASELINE], fx.metrics[CONTROLS]
    return _round(
        f"""
## Step 7: Round 1, controls

**What we changed:** a tight prompt, only the tools a page needs, slim tool results, and a strict JSON answer.

**What happened:** tool calls fall from {fx.value(BASELINE, "mean_tool_calls"):.1f} to {fx.value(CONTROLS, "mean_tool_calls"):.1f} per page, tokens from {fx.value(BASELINE, "mean_sync_tokens"):,.0f} to {fx.value(CONTROLS, "mean_sync_tokens"):,.0f}, model cost from ${fx.value(BASELINE, "model_cost_per_page_usd"):.5f} to ${fx.value(CONTROLS, "model_cost_per_page_usd"):.5f}. Personal-data leaks go from {before.pii_leaked_pages} pages to {after.pii_leaked_pages}.

**Why:** the baseline paid for work it never needed.
""",
        "round1",
        "BASELINE",
        "CONTROLS",
    )


def _round_two() -> list[Cell]:
    fx = facts()
    missed = fx.gates[ROUTING].failures.get("escalation_mismatch", 0)
    return _round(
        f"""
## Step 8: Round 2, model routing

**What we changed:** a cheap `gpt-6-luna` model reads each page first; only risky pages go to the stronger `gpt-6.1-terra`.

**What happened:** model cost falls to ${fx.value(ROUTING, "model_cost_per_page_usd"):.5f} per page, the lowest so far. The gate records {missed} missed attorney review{"" if missed == 1 else "s"} (guardrail pass {fx.gates[ROUTING].pass_rate:.0%}), but {missed} miss in {fx.pages} pages stays above the {fx.floor:.0%} floor, so Round 2 is {"accepted" if fx.gates[ROUTING].accepted else "rejected"}.

**Why:** a cheap first pass saves money and can slip. Cheaper is not free.
""",
        "round2",
        "CONTROLS",
        "ROUTING",
    )


def _round_three() -> list[Cell]:
    fx = facts()
    return [
        *_round(
            f"""
## Step 9: Round 3, a decision model

**What we changed:** one call to the OpenAI Decisions API asks the three questions and returns typed answers, a score and two yes/no probabilities, not free text. The router (plain code) picks the outcome. A cascade (a cheap model first, a stronger one only when unsure) lets `gpt-6.1-terra` re-check the unsure pages.

**What happened:** p50 latency falls from {fx.value(ROUTING, "p50_latency_s"):.2f}s to {fx.value(DECISION_MODEL, "p50_latency_s"):.2f}s and the missed attorney review is fixed, but model cost rises from ${fx.value(ROUTING, "model_cost_per_page_usd"):.5f} to ${fx.value(DECISION_MODEL, "model_cost_per_page_usd"):.5f} per page.

**Why:** typed answers are fast and can be calibrated (their thresholds tuned on part of the pages); the second opinion costs a little.
""",
            "round3",
            "ROUTING",
            "DECISION_MODEL",
        ),
        md("""
The router is plain code: readable, testable, auditable. A refusal or an error never produces a page; it fails safe to an attorney.
"""),
        code("""
import inspect

from doc_review_opt.agent.routing import decide

print(inspect.getsource(decide))
"""),
    ]


def _tradeoffs() -> list[Cell]:
    fx = facts()
    return [
        md(f"""
### Step 9b: LLM or Decisions?

Same three questions, four engines, {fx.pages} pages (simulated, calibrated thresholds). **Read it:** Decisions alone is the fastest and cheapest, but it sends more pages to an attorney ({fx.engine("decisions", "escalation_rate"):.0%} vs {fx.engine("llm-terra", "escalation_rate"):.0%} for `gpt-6.1-terra` as an LLM). That is why Round 3 adds the cascade.
"""),
        code(
            """
from doc_review_opt.experiment.tradeoffs import engine_comparison_table, plot_engine_tradeoff, simulate_engine_comparison

def show_engines(table):
    view = table[["label", "accuracy_vs_labels", "cost_per_page_usd", "p50_latency_s", "escalation_rate"]]
    view.columns = ["Engine", "Accuracy", "$ / page", "p50 (s)", "To attorney"]
    formats = {"Accuracy": "{:.0%}", "$ / page": "${:.5f}", "p50 (s)": "{:.2f}", "To attorney": "{:.0%}"}
    display(view.style.format(formats).hide(axis="index"))

engine_runs = await simulate_engine_comparison(policy=CALIBRATED_POLICY)
engines = engine_comparison_table(engine_runs)
show_engines(engines)
display(plot_engine_tradeoff(engines))
""",
            hidden=True,
        ),
        md("""
**Probabilities are a dial.** Move the privilege threshold: Decisions never leaks a privileged page; the LLMs' self-reported probabilities start leaking. Decisions cannot write text, extract spans or call tools, so each step gets the engine it needs.
"""),
        code(
            """
from doc_review_opt.experiment.tradeoffs import plot_threshold_sweep, threshold_sweep

display(plot_threshold_sweep(threshold_sweep(await simulate_engine_comparison(), EVAL_SET)))
""",
            hidden=True,
        ),
        code('draw_diagram("decisions_or_llm")', hidden=True),
    ]


def _round_four_five() -> list[Cell]:
    fx = facts()
    return [
        *_round(
            f"""
## Step 10: Round 4, prompt caching

**What we changed:** the unchanged start of every prompt (the review protocol) goes first, with one cache key per matter. Caching means the provider remembers that start, so resending it is billed at a lower rate.

**What happened:** model cost falls from ${fx.value(DECISION_MODEL, "model_cost_per_page_usd"):.5f} to ${fx.value(CACHING, "model_cost_per_page_usd"):.5f} per page.

**Why, and the catch:** the first call writes the cache at 1.25x the normal price, so caching pays only at volume.
""",
            "round4",
            "DECISION_MODEL",
            "CACHING",
        ),
        *_round(
            f"""
## Step 11: Round 5, split the workflow

**What we changed:** the privilege log, QA and tags leave the page path and run later as Batch jobs (a queue the provider runs within 24 hours at 50% off).

**What happened:** p50 latency falls from {fx.value(CACHING, "p50_latency_s"):.2f}s to {fx.value(SPLIT, "p50_latency_s"):.2f}s; model cost stays at about ${fx.value(SPLIT, "model_cost_per_page_usd"):.5f} per page.

**Why:** side jobs nobody waits for should not slow the page.
""",
            "round5",
            "CACHING",
            "SPLIT",
        ),
    ]


def _results() -> list[Cell]:
    fx = facts()
    default_cal, default_hold = (
        fx.half("default", "calibration"),
        fx.half("default", "holdout"),
    )
    refit_cal, refit_hold = fx.half("refit", "calibration"), fx.half("refit", "holdout")
    return [
        md("""
## Step 12: Results

Every round on one chart, then every metric. Hard gates are on top and decide acceptance; the rest only warn.
"""),
        code(
            """
floor = get_settings().quality.guardrail_pass_floor
display(plot_round_chart(summary))
display(plot_quality_gate(gate_table(traces, VARIANT_ORDER), floor))
""",
            hidden=True,
        ),
        code(
            """
table = metrics_table(traces, VARIANT_ORDER)
display(table.rename(columns=lambda name: name.replace("Round ", "R").split(":")[0]).style.hide(axis="index"))
display(confusion_table(traces, SPLIT))
""",
            hidden=True,
        ),
        code(
            """
from doc_review_opt.experiment.metrics import classifications_by_page, holdout_check

report = holdout_check(classifications_by_page(engine_runs["decisions"]))
display(holdout_table(report).style.hide(axis="index"))
print(f"Leak-free on the unseen half: {report.holds} ({report.calibration_pages} and {report.holdout_pages} pages)")
""",
            hidden=True,
        ),
        md(f"""
**Does calibration hold on unseen pages?** The policy is refit on one half of the pages (alternating within each route) and scored on the other. No leaks on unseen pages, and far cheaper than the default policy (default: {default_cal[0]:.0%} and {default_hold[0]:.0%} escalations). Escalations are {refit_cal[0]:.0%} on the calibration half and {refit_hold[0]:.0%} on the holdout half, and quality is {refit_cal[1]:.2f} and {refit_hold[1]:.2f}. Only the holdout rows are out of sample; {fx.holdout.holdout_pages} pages is a small sample.
"""),
    ]


def _live() -> list[Cell]:
    fx = facts()
    return [
        md(f"""
## Step 13: Live reality check

Real APIs, 2026-11-18, $0.18 total. The live run used 6 pages chosen from the original 30-page set, one per route in turn; the simulation above now uses all {fx.pages}.

| Engine | Accuracy | $ / page | p50 |
|---|---|---|---|
| **Decisions** | 100% | $0.00019 | **0.38s** |
| `gpt-6-luna` LLM | 100% | **$0.00011** | 2.26s |
| `gpt-6.1-terra` LLM | 100% | $0.00073 | 3.03s |
| `gpt-6.1-terra` agent | 100% | $0.00118 | 2.33s |

Decisions is **about 6x faster** than `gpt-6-luna` and **4 to 6x cheaper** than `gpt-6.1-terra`, but **1.7x pricier** than `gpt-6-luna`.
"""),
        code(
            """
from doc_review_opt.reporting.reporting import plot_sim_vs_live, recorded_live_note, recorded_live_summary

display(plot_sim_vs_live(summary, recorded_live_summary(), recorded_live_note()))
""",
            hidden=True,
        ),
        md(f"""
**Reading the live chart.** Live Round 2 was slower than Round 1 ({fx.live(ROUTING).p50_latency_s:.2f}s vs {fx.live(CONTROLS).p50_latency_s:.2f}s) and scored lower (quality {fx.live(ROUTING).quality:.2f} vs {fx.live(CONTROLS).quality:.2f}). Live Decisions scored the same as Round 1 ({fx.live(DECISION_MODEL).quality:.2f}) on its {fx.live_run.pages} pages, while the {fx.pages}-page simulation shows {fx.engine("decisions", "accuracy_vs_labels"):.0%} route accuracy for Decisions alone. Live Round 4 caching cost more than Round 3 (${fx.live(CACHING).model_cost_per_page_usd:.5f} vs ${fx.live(DECISION_MODEL).model_cost_per_page_usd:.5f} per page).
"""),
        md("""
Optional, on stage: rerun the engine comparison on the real APIs. It takes about a minute, costs a few cents and sends LangSmith traces.
"""),
        code(
            """
from doc_review_opt.experiment.live import pick_live_pages
from doc_review_opt.experiment.tradeoffs import live_engine_comparison

if not RUN_LIVE_API_CALLS:
    print("Offline. Set RUN_LIVE_API_CALLS=true to rerun this on the real APIs.")
else:
    live_runs = await live_engine_comparison(pick_live_pages(EVAL_SET), client=client, budget=budget)
    if live_runs:
        show_engines(engine_comparison_table(live_runs))
        print("\\n".join(f"{n}: {r.trace_url or 'LangSmith tracing is off'}" for n, r in live_runs.items()))
    print(f"Spend: ${budget.total_usd:.4f} of ${budget.limit_usd:.2f}")
""",
            hidden=True,
        ),
    ]


def _closing() -> list[Cell]:
    fx = facts()
    return [
        md(f"""
## Step 14: Takeaways

1. **Measure first.** A labelled eval set and hard gates before any tuning.
2. **Cut waste before changing models.** Controls gave the biggest drop.
3. **Cheapest is not best.** Check what each round gives up.
4. **Typed decision? Try Decisions with a code router.** Fast, calibratable, auditable.
5. **Need text, tools or extraction? Keep an LLM**, the smallest that holds quality.
6. **Re-measure live.** In the live engine comparison, {ENGINE_SHORT[fx.live_fastest().engine]} was the fastest engine ({fx.live_fastest().p50_latency_s:.2f}s) and {ENGINE_SHORT[fx.live_cheapest().engine]} the cheapest (${fx.live_cheapest().model_cost_per_page_usd:.5f} per page); in the live rounds, routing was slower and caching cost more.
"""),
        md("""
## References

Last verified: November 18, 2026.

- [Optimizing agents for cost and quality](https://github.com/openai/openai-cookbook/tree/main/examples/agent_optimization) (OpenAI Cookbook)
- [Document review agent](https://gist.github.com/sydney-runkle/a632ba4ea0b2b72501dfa4b6ab2a7d8a) (gist by Sydney Runkle)
- [Decisions API guide](https://developers.openai.com/api/docs/guides/decisions)
- [Pricing](https://developers.openai.com/api/docs/pricing)
- LangGraph [`Send`](https://langchain-ai.github.io/langgraph/concepts/low_level/#send) and [`interrupt`](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
"""),
    ]


def cells() -> list[Cell]:
    return [
        *_intro(),
        *_system(),
        *_optimize(),
        *_test_set(),
        *_baseline(),
        *_follow(),
        *_examples(),
        *_round_one(),
        *_round_two(),
        *_round_three(),
        *_tradeoffs(),
        *_round_four_five(),
        *_results(),
        *_live(),
        *_closing(),
    ]

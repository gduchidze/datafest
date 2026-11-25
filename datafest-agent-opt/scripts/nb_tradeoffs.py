"""Notebook cells: decision models vs LLMs, scenario mapping, guardrails and the live run."""

# ruff: noqa: E501
from __future__ import annotations

from doc_review_opt.fixtures import EVAL_SET
from scripts.cells import Cell, code, md

PAGES = len(EVAL_SET)


def _engines() -> list[Cell]:
    return [
        md("""
## Tradeoffs and scenario mapping

This section zooms out. It asks **when a decision model is better and when a step still needs an LLM**, using a step matrix, an engine comparison, a threshold sweep and cascade economics. Then it maps architectures to operating scenarios.

### Decisions or LLM?

What to look at: start at the top and answer in order; the first yes decides the engine.
"""),
        code("""
draw_diagram("decisions_or_llm")
"""),
        md("""
The matrix goes through every step of the agent and names the engine that fits.
"""),
        code("""
from doc_review_opt.experiment.tradeoffs import STEP_MATRIX, WHEN_TO_USE

display(pd.DataFrame(STEP_MATRIX))
"""),
        md(f"""
**Engine comparison.** The same {PAGES} pages, graph and router, with only the classifier swapped: Decisions, the same three questions on `gpt-6-luna` and on `gpt-6.1-terra` as LLMs, and the controlled agent. Accuracy is route accuracy against labels; agreement is with Decisions. The first table uses the default thresholds, the second the thresholds calibrated for Decisions. Note what moves when only thresholds change.
"""),
        code("""
tables = []
for name, runs in (("default", engine_runs_default), ("calibrated", engine_runs_calibrated)):
    tables.append(engine_comparison_table(runs).drop(columns="label").assign(policy=name))
both_policies = pd.concat(tables, ignore_index=True)
display(
    both_policies[
        [
            "policy",
            "engine",
            "accuracy_vs_labels",
            "agreement_vs_decisions",
            "cost_per_100k_pages_usd",
            "p50_latency_s",
            "escalation_rate",
            "privileged_leaks",
            "pii_leaks",
        ]
    ].style.format(ENGINE_FORMATS, na_rep="-")
)
"""),
        md("""
**Live engine comparison (optional).** With `RUN_LIVE_API_CALLS=true` the four engines run on real APIs over `LIVE_PAGE_LIMIT` pages. With `LANGSMITH_TRACING=true` each run is one trace named `doc_review::<engine>`. The $1 budget still applies.
"""),
        code("""
from doc_review_opt.experiment.tradeoffs import live_engine_comparison, plot_engine_tradeoff

live_engine_runs = {}
if RUN_LIVE_API_CALLS:
    try:
        live_engine_runs = await live_engine_comparison(
            LIVE_PAGES, client=client, budget=budget
        )
    except BudgetExceededError as error:
        print("Stopped by the budget:", error)
    if live_engine_runs:
        live_engine_table = engine_comparison_table(live_engine_runs)
        display(live_engine_table.drop(columns="label").style.format(ENGINE_FORMATS, na_rep="-"))
        for name, run in live_engine_runs.items():
            print(f"{name}: {run.trace_url or 'LangSmith tracing is off'}")
    print(f"Running spend: ${budget.total_usd:.4f} of ${budget.limit_usd:.2f}")
else:
    print("Dry-run mode. Set OPENAI_API_KEY and RUN_LIVE_API_CALLS=true to compare the engines on the real APIs.")
"""),
        md("""
**Cost against latency.** One dot per engine; the bottom-left corner is better. In the simulation, Decisions is cheap and fast, and the strongest LLM costs roughly ten times more per page (accuracy is in the table above). The Decisions profile is an assumption, so read the shape, not the exact values. Live (2026-11-18, 6 pages), Decisions was fastest but not cheapest: `gpt-6-luna` as an LLM cost about 0.6x as much per page.
"""),
        code("""
display(plot_engine_tradeoff(engine_comparison_table(engine_runs_calibrated)))
"""),
        md("""
**Threshold sweep.** A lower privilege threshold escalates more and leaks fewer privileged pages. Each engine needs its own operating point. **How to read it:** each line is one engine; a red cross marks a threshold where a privileged page would be produced.
"""),
        code("""
from doc_review_opt.experiment.tradeoffs import plot_threshold_sweep, threshold_sweep

sweep = threshold_sweep(engine_runs_default, EVAL_SET)
for column, fmt in (("escalation_rate", "{:.0%}"), ("privileged_leaks", "{:.0f}")):
    print(column)
    display(
        sweep.pivot(index="privilege_threshold", columns="engine", values=column)
        .style.format(fmt)
        .format_index("{:.2f}")
    )
display(plot_threshold_sweep(sweep))
"""),
        md("""
**Cascade economics (simulated).** Decisions alone has the lowest model cost, but its extra mistakes cost attorney time. A `gpt-6.1-terra` second opinion on the ambiguous band costs a little more and cuts escalations. `gpt-6.1-terra` on everything costs several times more for no better quality. `blended` includes attorney time.
"""),
        code("""
from doc_review_opt.experiment.tradeoffs import cascade_analysis

cascade = await cascade_analysis()
display(
    cascade.style.format(
        {
            "model_cost_per_page_usd": "${:.6f}",
            "attorney_cost_per_page_usd": "${:.3f}",
            "blended_cost_per_page_usd": "${:.3f}",
            "mean_quality": "{:.2f}",
        }
    )
)
"""),
        md("""
**Where Decisions fits and where an LLM is needed.**
"""),
        code("""
display(pd.DataFrame(WHEN_TO_USE))
"""),
    ]


def _scenarios() -> list[Cell]:
    return [
        md("""
### Candidate architectures

Each row turns a sprint outcome into an architecture: models, tools, caching, and whether slow work is split off.
"""),
        code("""
from doc_review_opt.experiment.scenarios import (
    ARCHITECTURE_OPTIONS,
    OPERATING_SCENARIOS,
    best_fit,
    scenario_map,
)

display(pd.DataFrame(ARCHITECTURE_OPTIONS).T)
"""),
        md("""
### Scenario sweet spots

Scenarios differ in what they cannot compromise on: quality, policy, latency, budget, escalation rate, async needs and traffic repeat (caching needs high repeat). The fit score adds or subtracts points per constraint:

- quality +3 / -4, policy +3 / -5
- latency and budget +2 / -2
- async +2 / -1, escalation rate +2 / -2
- a cache-locality penalty, and +1 if recommended

The table shows the best-scoring architecture per scenario.
"""),
        code("""
display(
    pd.DataFrame(OPERATING_SCENARIOS)[
        [
            "scenario",
            "description",
            "quality_floor",
            "policy_floor",
            "p50_latency_target_s",
            "monthly_budget_100k_pages_usd",
            "max_escalation_rate",
            "needs_async",
            "cache_locality",
        ]
    ]
)
SCENARIO_FORMATS = {
    "quality": "{:.2f}",
    "policy_compliance": "{:.0%}",
    "escalation_rate": "{:.0%}",
    "p50_latency_s": "{:.2f}",
    "cost_per_page_usd": "${:.5f}",
    "monthly_cost_at_100k_pages": "${:,.0f}",
}
display(best_fit(full_summary).reset_index(drop=True).style.format(SCENARIO_FORMATS))
"""),
        md("""
### Full combination map for one scenario

Every architecture scored against one scenario, so you can see which constraint each loser fails. This one is the second-request deadline: huge volume, a court deadline, and cost per page matters most.
"""),
        code("""
deadline = scenario_map(full_summary)
deadline = deadline[deadline["scenario"] == OPERATING_SCENARIOS[0]["scenario"]]
display(
    deadline.sort_values("score", ascending=False)
    .drop(columns="scenario")
    .reset_index(drop=True)
    .style.format(SCENARIO_FORMATS)
)
"""),
    ]


def _monitoring() -> list[Cell]:
    return [
        md("""
## Monitoring, evals and guardrails

Deterministic guardrails run on every trace and need no model. They check:

- **hard gates:** a privileged page produced, personal data in produced text
- **outcome:** responsive page withheld, non-responsive page produced, escalation mismatch, unnecessary attorney review
- **interface:** page text in an interrupt payload, unparseable reply, unescalated classifier error
- **efficiency:** too many unnecessary tool calls
"""),
        code("""
from doc_review_opt.experiment.evaluation import evaluate_answer_traces, summarize_answer_evals

main_traces = [trace for trace in traces if trace.variant in VARIANT_ORDER]
guard_results = await evaluate_answer_traces(EVAL_SET, main_traces)
guard_summary = guard_results.groupby("variant_label", sort=False).agg(
    pages=("doc_id", "count"),
    deterministic_pass_rate=("deterministic_passed", "mean"),
)
display(guard_summary.style.format({"deterministic_pass_rate": "{:.0%}"}))
failing = guard_results[~guard_results["deterministic_passed"]]
print(f"{len(failing)} of {len(guard_results)} traces fail at least one check. Failing rows, first 12:")
display(failing[["variant_label", "doc_id", "deterministic_failures"]].head(12))
"""),
        md("""
### Optional: LLM judge

The judge (`gpt-6.1-terra`, strict `{passed, reason}` schema, no retries) grades whether the outcome is defensible. It never overrides a deterministic failure: a trace passes only if **both** pass. It runs on the simulated traces for `LIVE_PAGE_LIMIT` pages across six variants, charged to the $1 budget. Without `RUN_LLM_JUDGE=true` and `OPENAI_API_KEY`, the summary shows only the deterministic pass rate.
"""),
        code("""
judge_ids = {page.doc_id for page in LIVE_PAGES}
if RUN_LLM_JUDGE:
    judge_traces = [trace for trace in main_traces if trace.doc_id in judge_ids]
    try:
        judge_results = await evaluate_answer_traces(
            EVAL_SET, judge_traces, client=judge_client, budget=budget
        )
    except BudgetExceededError as error:
        print("Stopped by the budget:", error)
        judge_results = guard_results
    print(f"Judged {len(judge_traces)} traces. Running spend: ${budget.total_usd:.4f} of ${budget.limit_usd:.2f}")
else:
    judge_results = guard_results
    print("Dry-run mode. Set OPENAI_API_KEY and RUN_LLM_JUDGE=true to grade the simulated traces with the judge.")
display(
    summarize_answer_evals(judge_results).style.format(
        {
            "deterministic_pass_rate": "{:.0%}",
            "judge_coverage": "{:.0%}",
            "judge_pass_rate": "{:.0%}",
            "both_pass_rate": "{:.0%}",
            "known_judge_cost_usd": "${:.4f}",
        },
        na_rep="-",
    )
)
"""),
    ]


def _full_live() -> list[Cell]:
    return [
        md("""
## Optional: full live run

With `RUN_LIVE_API_CALLS=true` every variant runs on the real APIs over the same `LIVE_PAGE_LIMIT` pages, with async fan-out and the shared budget. The graph, router and checks match the simulation, so the same summary applies. It stops cleanly at the $1 cap. With `LANGSMITH_TRACING=true` each variant is one trace and the URLs are printed.

Live numbers differ from the simulation because latency, tokens, caching and accuracy are measured. On a few pages they are anecdotes, not a benchmark.
"""),
        code("""
live_runs = {}
if RUN_LIVE_API_CALLS:
    for variant in VARIANT_ORDER:
        try:
            live_runs[variant] = await run_live_variant(
                variant, LIVE_PAGES, client=client, budget=budget
            )
        except BudgetExceededError as error:
            print(f"Stopped before finishing {variant}: {error}")
            break
        print(f"{variant}: spend so far ${budget.total_usd:.4f} of ${budget.limit_usd:.2f}")
    if live_runs:
        live_traces = [trace for run in live_runs.values() for trace in run.traces]
        live_summary = summarize(traces_frame(live_traces))
        show(
            live_summary[
                [
                    "variant_label",
                    "pages",
                    "mean_quality",
                    "route_accuracy",
                    "escalation_rate",
                    "privileged_leaks",
                    "pii_leaks",
                    "mean_sync_tokens",
                    "p50_latency_s",
                    "model_cost_per_page_usd",
                ]
            ]
        )
        display(pd.DataFrame([{"variant": v, "trace_url": run.trace_url} for v, run in live_runs.items()]))
else:
    print("Dry-run mode. Set OPENAI_API_KEY and RUN_LIVE_API_CALLS=true to run every variant on the real APIs.")
"""),
        md("""
## What the live run showed

Live run, 2026-11-18, 6 pages (`LIVE_PAGE_LIMIT`), $0.18 of the $1 cap, no errors. Measured, not modeled, and six pages are an anecdote. Anything above not labelled live is simulated.

| Round | Quality | Escalation | Sync tokens | p50 (s) | Model $/page |
|---|---|---|---|---|---|
| Bad baseline | 0.84 | 33% | 16,996 | 10.92 | 0.02243 |
| R1 controls | 0.98 | 17% | 942 | 2.96 | 0.00112 |
| R2 routing | 0.95 | 17% | 1,326 | 4.00 | 0.00039 |
| R3 decision model | 0.98 | 17% | 1,772 | 0.45 | 0.00008 |
| R4 caching | 0.98 | 17% | 1,772 | 0.43 | 0.00021 |
| R5 split | 0.98 | 17% | 1,772 | 0.34 | 0.00008 |

Zero privileged or PII leaks in every round. Judge (`gpt-6.1-terra`): the baseline passed the judge on 83% of pages but the deterministic checks on 0%; Rounds 1 to 5 passed both on 100%.

**Live engine comparison** (same 6 pages):

| Engine | Accuracy | Escalation | Model $/page | p50 (s) |
|---|---|---|---|---|
| Decisions | 100% | 17% | 0.00019 | 0.38 |
| `gpt-6-luna` as an LLM | 100% | 17% | 0.00011 | 2.26 |
| `gpt-6.1-terra` as an LLM | 100% | 17% | 0.00073 | 3.03 |
| Controlled agent (`gpt-6.1-terra`) | 100% | 17% | 0.00118 | 2.33 |

Separate live probe (none of the 6 pages above was attorney-cc'd): Decisions privileged probability on the two original attorney-cc'd mails (original wording / protocol wording): ARC-000306 0.02 / 0.00; ARC-000413 0.00 / 0.00.

**Where live diverged from the simulation.**

1. **Round 2 was slower, not faster.** p50 4.00s against 2.96s, and lower quality (0.95 against 0.98). Model cost per page still fell ($0.00039 against $0.00112).
2. **Decisions won on latency, not always price.** About 6x faster but 1.7x pricier than `gpt-6-luna` as an LLM; 4 to 6x cheaper and 6 to 8x faster than `gpt-6.1-terra`. Its other benefits are calibratable probabilities and an auditable router.
3. **The wording lever was not needed.** The real API scores cc'd mail near 0 with either wording, so the simulated over-escalation (about 0.3) did not reproduce. The protocol wording stays because it is free.
4. **Round 4 caching cost more** on 6 pages ($0.00021 against $0.00008 per page). It pays only at volume.
"""),
        md("""
**How to read it.** Blue is the simulation, orange is the measured live run. The recorded values are the table above; a live session in this notebook replaces them.
"""),
        code("""
from doc_review_opt.reporting.reporting import plot_sim_vs_live, recorded_live_note, recorded_live_summary

if live_runs:
    live_view, live_note = live_summary, f"live: {LIVE_PAGE_LIMIT} pages, this session"
else:
    live_view, live_note = recorded_live_summary(), recorded_live_note()
display(plot_sim_vs_live(summary, live_view, live_note))
"""),
    ]


def cells() -> list[Cell]:
    return [*_engines(), *_scenarios(), *_monitoring(), *_full_live()]

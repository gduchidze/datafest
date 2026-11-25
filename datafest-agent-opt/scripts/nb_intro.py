"""Notebook cells: title, setup, simulation, summary and the round helpers."""

# ruff: noqa: E501
from __future__ import annotations

from doc_review_opt.fixtures import EVAL_SET
from scripts.cells import Cell, code, md

PAGES = len(EVAL_SET)


def _opening() -> list[Cell]:
    return [
        md("""
# Optimizing document review agents for cost and quality: decision models vs LLMs

This notebook measures a baseline document review agent, changes one thing per round, and accepts the savings only if quality holds. It also asks which steps should be a decision model and which still need an LLM. The default run is an offline simulation on synthetic pages; optional live cells use real APIs under a $1 budget.

### The optimization loop

What to look at: the dashed arrow, which is why we never stop after one round.
"""),
        code("""
from doc_review_opt.reporting.diagrams import draw_diagram

draw_diagram("optimization_loop")
"""),
        md("""
## Outline

1. Success criteria and a small eval set.
2. An intentionally inefficient baseline agent.
3. Baseline cost, tokens, quality, latency, tool calls and leaks.
4. Round 1: prompt, output, tool and context controls.
5. Round 2: route routine pages to a smaller model.
6. Round 3: a decision model (Decisions API) plus a code router.
7. Round 4: prompt caching.
8. Round 5: split real-time and background work.
9. Human-in-the-loop attorney review.
10. Tradeoffs and operating scenarios, including decision models vs LLMs.
11. Monitoring, evals and guardrails.
"""),
        md("""
## Use case

Fictional matter: *Hollis v. Arcadia Freight*. The agent sees one page and picks one outcome: **produce**, **withhold**, **redact** or **attorney review**.

Never acceptable: a **privileged** page produced, or **personal data** left in produced text.

Five mock tools stand in for real systems: custodian directory, review protocol, prior productions, redaction and attorney escalation. The baseline is one `gpt-6.1-terra` agent that does everything, expensively, on purpose.

Models: `gpt-6.1-terra` (strongest tier), `gpt-6-luna` (small tier) and the Decisions API, which runs on `gpt-6-luna`. Reasoning is off for every Responses call (`reasoning={"effort": "none"}`), so it is not a knob here.
"""),
        md(
            """
## Setup

Use Python 3.13, then install and open the notebook:

```bash
uv sync --group notebook
uv run jupyter lab
```

Nothing calls an API by default. To opt in, set these in `.env` (git-ignored) or the environment:

```bash
export OPENAI_API_KEY=...
export LANGSMITH_API_KEY=...        # only for tracing
export RUN_LIVE_API_CALLS=true      # live agent, Decisions and engine-comparison cells
export RUN_LLM_JUDGE=true           # optional answer judge (gpt-6.1-terra)
export LANGSMITH_TRACING=true       # send traces for the live runs to LangSmith
```

Environment values win over `.env`. All live cells share one `BudgetTracker` with a hard $1 stop (limit in `config.toml`). Leave the flags unset for a fully offline run. The Batch example only writes a local file under `outputs/`; it never submits.

### How the notebook runs

What to look at: both paths fill the same trace table, so every table and chart below works either way.
"""
        ),
        code("""
draw_diagram("notebook_modes")
"""),
        code("""
%matplotlib inline
import inspect
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import AsyncOpenAI

load_dotenv(override=False)


def env_flag(name):
    return os.environ.get(name, "false").lower() == "true"


RUN_LIVE_API_CALLS = env_flag("RUN_LIVE_API_CALLS")
RUN_LLM_JUDGE = env_flag("RUN_LLM_JUDGE")
LANGSMITH_TRACING = env_flag("LANGSMITH_TRACING")
if (RUN_LIVE_API_CALLS or RUN_LLM_JUDGE) and not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY before enabling the live agent or judge.")
if LANGSMITH_TRACING and not os.environ.get("LANGSMITH_API_KEY"):
    raise RuntimeError("Set LANGSMITH_API_KEY before enabling LangSmith tracing.")
os.environ["LANGSMITH_TRACING"] = str(LANGSMITH_TRACING).lower()
os.environ["LANGCHAIN_TRACING_V2"] = str(LANGSMITH_TRACING).lower()
os.environ.setdefault("LANGSMITH_PROJECT", "doc-review-opt")

from doc_review_opt.config import get_settings
from doc_review_opt.experiment.live import LIVE_PAGE_LIMIT, BudgetTracker

LIVE = get_settings().live
client = AsyncOpenAI(timeout=LIVE.timeout_s, max_retries=LIVE.max_retries) if RUN_LIVE_API_CALLS else None
judge_client = AsyncOpenAI(timeout=LIVE.timeout_s, max_retries=LIVE.max_retries) if RUN_LLM_JUDGE else None
budget = BudgetTracker()
OUTPUT_DIR = Path(os.environ.get("DOC_REVIEW_OUTPUT_DIR", "../outputs"))

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", None)
print("RUN_LIVE_API_CALLS =", RUN_LIVE_API_CALLS)
print("RUN_LLM_JUDGE =", RUN_LLM_JUDGE)
print("LANGSMITH_TRACING =", LANGSMITH_TRACING)
print(f"Live budget hard stop: ${budget.limit_usd:.2f}; live page limit: {LIVE_PAGE_LIMIT}")
"""),
        md(f"""
## Success criteria

Accept savings only if quality holds. The gates, in order:

1. **Hard gates:** no privileged page produced, no personal data left in produced text.
2. **Quality:** the right outcome per page.
3. **Escalation accuracy:** attorney review only where needed.
4. **Cost and latency:** compared only after the gates pass. Cost includes attorney time.

The eval set has {PAGES} pages, split into a calibration half and a holdout half. Thresholds are tuned on the calibration half only.
"""),
        md("""
## Simulation contract

The default path uses synthetic pages and modeled metrics. Its numbers are **not a production benchmark**.

- **Real:** the LangGraph review graph, the router and the guardrail checks.
- **Modeled:** token counts, latency, cache hits and quality score, from illustrative formulas and the prices in `config.toml`.
- **Assumed:** each engine's *ambiguity profile* (how sharp its probabilities are and where it errs).
- **Fixture-labelled:** tool selection, so tool choice is not measured.

The optional live cells produce the same trace schema from real API calls.
"""),
        md("""
## Optimization knobs

| Knob | Inefficient baseline | Optimized pattern | Primary metric |
|---|---|---|---|
| Prompt and output | "Be thorough" instructions, `verbosity="high"`, free-text answer | Specific rules, a strict `json_schema` disposition, `verbosity="low"`, capped output | Output tokens, parse failures |
| Tool surface | All five tools exposed, `tool_choice="required"` | Full stable tool list plus `allowed_tools` per page type | Tool calls, unnecessary tools |
| Tool schemas and payloads | Verbose descriptions, audit logs and directories in every result | Small schemas and slim payloads with only decision-critical fields | Input tokens |
| Model routing | One `gpt-6.1-terra` agent for every page | `gpt-6-luna` triage and routine pages; `gpt-6.1-terra` for privileged and borderline pages | Cost, latency, escalation accuracy |
| **Decisions vs LLM** | An LLM agent decides every page, then explains itself | One Decisions call (score `responsive`, predicates `privileged` and `contains_pii`) plus a code router; an LLM only for redaction text and a second opinion on the ambiguous band | Cost, latency, typed routing, calibration effort |
| Prompt caching | Volatile page text mixed into the prefix | Stable protocol first, page last, a `prompt_cache_key` per matter | Cached input tokens, cost |
| Workflow split | Privilege-log draft, QA and production tags in the review path | Review path sync; the rest as Batch jobs at 50% off | p50 latency, synchronous cost |
| Guardrails and evals | Informal spot checks | Deterministic checks plus a judge schema for live traces | Regression rate, safety pass rate |

Reasoning is off everywhere, so it is not a row here.
"""),
        md(f"""
## Sample evaluation set

{PAGES} synthetic pages: the six original pages plus borderline responsiveness, attorney-cc'd business mail, several personal-data kinds, OCR noise and near-duplicates. Each page is labelled for responsiveness (0 to 2), privilege, personal data, expected route, escalation, difficulty and risk.
"""),
        code("""
from doc_review_opt.fixtures import CALIBRATION, EVAL_SET, HOLDOUT, MATTER

eval_table = pd.DataFrame(
    {
        "doc_id": page.doc_id,
        "custodian": page.custodian,
        "difficulty": page.labels.difficulty,
        "risk": page.labels.risk,
        "responsive_level": page.labels.responsive_level,
        "privileged": page.labels.privileged,
        "contains_pii": page.labels.contains_pii,
        "expected_route": page.labels.expected_route,
        "must_escalate": page.labels.must_escalate,
        "expected_tools": ", ".join(page.labels.expected_tools),
    }
    for page in EVAL_SET
)
print(f"{MATTER}: {len(EVAL_SET)} pages ({len(CALIBRATION)} calibration, {len(HOLDOUT)} holdout)")
display(eval_table.head(10))
display(eval_table.groupby(["expected_route", "difficulty"]).size().rename("pages").to_frame())
"""),
        md("""
**How to read it.** Left: how many pages each route should end in. Right: how hard the pages are.
"""),
        code("""
from doc_review_opt.reporting.reporting import plot_eval_set

display(plot_eval_set(eval_table))
"""),
        md("""
## Support data and tools

Five local functions in `agent/tools.py` stand in for the matter's systems. The baseline returns oversized payloads; later rounds return only decision-critical fields. The next cell shows one slim custodian record.
"""),
        code("""
from doc_review_opt.agent.tools import lookup_custodian
from doc_review_opt.fixtures import ATTORNEYS

print(json.dumps(lookup_custodian(ATTORNEYS[0], payload="slim"), indent=2))
"""),
        code("""
from doc_review_opt.agent.tools import SLIM_TOOLS, VERBOSE_TOOLS
from doc_review_opt.pricing import approx_tokens

print("Verbose custodian payload tokens:", approx_tokens(lookup_custodian(ATTORNEYS[0], payload="verbose")))
print("Slim custodian payload tokens:", approx_tokens(lookup_custodian(ATTORNEYS[0], payload="slim")))
print("Verbose tool schema tokens:", approx_tokens(VERBOSE_TOOLS))
print("Slim tool schema tokens:", approx_tokens(SLIM_TOOLS))
"""),
        md(
            """
## Baseline architecture

The baseline does too much in one synchronous path.

What to look at: the single box doing every job; that is the cost and latency we will take apart.
"""
        ),
        code("""
draw_diagram("bad_baseline")
"""),
        md("""
### The review graph

Every variant runs through the same LangGraph graph. A `Send` fans pages out to one subgraph each; every page is classified, routed in code and ends in one terminal node. `attorney_review` pauses with `interrupt()`. Only IDs and scores cross that boundary, never page text.

What to look at: follow one page; three typed answers go in, plain code picks one of four outcomes.
"""),
        code("""
draw_diagram("page_review")
"""),
        code("""
from doc_review_opt.agent.graph import build_review_graph, review_checkpointer
from doc_review_opt.agent.routing import RoutingPolicy
from doc_review_opt.agent.variants import CALIBRATED_POLICY, DECISION_MODEL, VARIANTS
from doc_review_opt.experiment.simulation import build_simulated_engines

graph_classifier, graph_redactor = build_simulated_engines(
    VARIANTS[DECISION_MODEL], EVAL_SET, CALIBRATED_POLICY
)
review_graph = build_review_graph(
    graph_classifier, graph_redactor, review_checkpointer(), policy=CALIBRATED_POLICY
)
print("Graph nodes:", ", ".join(review_graph.get_graph(xray=True).nodes))
"""),
        md("""
## Metrics helpers

The helpers read token usage from each response, including cached tokens. Output tokens already include reasoning tokens (none here).

Prices below are USD per million tokens, verified November 18, 2026. Decisions has the same input price as `gpt-6-luna`, **Decisions output is not charged**, and Batch is 50% off. Long-context and priority pricing are not covered.
"""),
        code("""
from doc_review_opt.config import get_settings
from doc_review_opt.pricing import BATCH_DISCOUNT, MODEL_PRICES_USD_PER_1M

display(pd.DataFrame(MODEL_PRICES_USD_PER_1M).T.rename_axis("model"))
print(f"Batch multiplier: {BATCH_DISCOUNT} (50% off)")
print(f"Attorney review assumption: ${get_settings().attorney.review_cost_usd_per_page:.2f} per escalated page")
"""),
        md(f"""
## Dry-run simulation

The [simulation](../src/doc_review_opt/experiment/simulation.py) runs each variant on the same {PAGES} pages through the review graph with simulated engines, so routing can really err. Cost is model cost plus attorney time per escalated page.

- Decision-model rounds use thresholds calibrated on the calibration half.
- Caching assumes a warm cache and the 1,024-token minimum; it is not measured.
- Round 5 removes background work from synchronous latency but still counts its cost.
- `03a_router_only` (Decisions plus code router, no second opinion) is the "Decisions only" arm.
"""),
        code("""
from doc_review_opt.agent.variants import EXTRA_VARIANTS, VARIANT_ORDER
from doc_review_opt.experiment.simulation import simulate_all
from doc_review_opt.experiment.traces import traces_frame

traces = await simulate_all(include_extras=True)
frame = traces_frame(traces)
frame.drop(columns="produced_text").head()
"""),
        md("""
## Summary

One row per variant. `blended_cost_per_verified_page` is model cost plus attorney cost, divided by correctly dispositioned pages.
"""),
        code("""
from doc_review_opt.reporting.reporting import summarize

full_summary = summarize(frame)
summary = full_summary[full_summary["variant"].isin(VARIANT_ORDER)].reset_index(drop=True)

USD_FORMATS = {
    "model_cost_per_page_usd": "${:.5f}",
    "cost_per_page_usd": "${:.3f}",
    "blended_cost_per_verified_page": "${:.3f}",
    "monthly_model_cost_at_100k_pages": "${:,.0f}",
    "monthly_cost_at_100k_pages": "${:,.0f}",
}
FORMATS = {
    "mean_quality": "{:.2f}",
    "policy_compliance": "{:.0%}",
    "route_accuracy": "{:.0%}",
    "escalation_accuracy": "{:.0%}",
    "escalation_rate": "{:.0%}",
    "privileged_leaks": "{:d}",
    "pii_leaks": "{:d}",
    "mean_tool_calls": "{:.1f}",
    "mean_extra_tool_calls": "{:.1f}",
    "mean_sync_tokens": "{:,.0f}",
    "mean_total_tokens": "{:,.0f}",
    "mean_cached_tokens": "{:,.0f}",
    "p50_latency_s": "{:.2f}",
    "sync_cost_per_page_usd": "${:.6f}",
    "background_cost_per_page_usd": "${:.6f}",
    "cost_reduction_vs_baseline": "{:.0%}",
    "mean_sync_tokens_delta_vs_previous": "{:+,.0f}",
    "mean_total_tokens_delta_vs_previous": "{:+,.0f}",
    "p50_latency_s_delta_vs_previous": "{:+.2f}",
    "model_cost_per_page_usd_delta_vs_previous": "${:+.5f}",
    "cost_per_page_usd_delta_vs_previous": "${:+.3f}",
    "quality_delta_vs_previous": "{:+.2f}",
    **USD_FORMATS,
}


def show(table):
    display(table.style.format(FORMATS, na_rep="-"))


show(
    summary[
        [
            "variant_label",
            "mean_quality",
            "policy_compliance",
            "route_accuracy",
            "escalation_rate",
            "privileged_leaks",
            "pii_leaks",
            "mean_tool_calls",
            "mean_sync_tokens",
            "mean_total_tokens",
            "mean_cached_tokens",
            "p50_latency_s",
            "model_cost_per_page_usd",
            "cost_per_page_usd",
            "blended_cost_per_verified_page",
            "cost_reduction_vs_baseline",
            "monthly_model_cost_at_100k_pages",
        ]
    ]
)
"""),
        md("""
## Round-by-round impact

Each row compares a round with the one before. `model_cost` is the API bill; `cost_per_page` adds attorney time.
"""),
        code("""
from doc_review_opt.reporting.reporting import round_impact

impact = round_impact(summary)
show(impact)
"""),
        md("""
## Cost, latency and quality

**How to read it.** Left: tokens per page. Middle: model cost per page on a log scale. Right: quality. Simulated.
"""),
        code("""
from doc_review_opt.reporting.reporting import plot_round_chart

display(plot_round_chart(summary))
"""),
        md("""
**How to read it.** Left: model cost per page, with the change from the previous round. Right: median latency.
"""),
        code("""
from doc_review_opt.reporting.reporting import plot_cost_latency

display(plot_cost_latency(summary))
"""),
        md("""
### Where the money goes

**How to read it.** Orange is attorney time, blue is the model bill. The model bill is the sliver at the bottom.
"""),
        code("""
from doc_review_opt.reporting.reporting import plot_cost_split

display(plot_cost_split(summary))
"""),
    ]


def _helpers() -> list[Cell]:
    return [
        md("""
## Round helpers

Every round has six beats: **symptom**, **lever**, **diff**, **re-measure**, **quality gate** and **tradeoff**. The helpers below keep them short.
"""),
        code("""
from collections import Counter

from doc_review_opt.agent.variants import (
    BASELINE,
    CACHING,
    CONTROLS,
    ORIGINAL_WORDING_VARIANT,
    ROUTER_ONLY,
    ROUTING,
    SPLIT,
    VARIANT_LABELS,
)
from doc_review_opt.experiment.simulation import variant_request_example
from doc_review_opt.experiment.traces import deterministic_guardrail_check
from doc_review_opt.reporting.reporting import per_page_view, plot_progress

GUARDRAIL_FLOOR = get_settings().quality.guardrail_pass_floor
PAGE_BY_ID = {page.doc_id: page for page in EVAL_SET}
SUMMARY_BY_VARIANT = full_summary.set_index("variant")
TRACES = {
    variant: [t for t in traces if t.variant == variant]
    for variant in VARIANT_ORDER + EXTRA_VARIANTS
}
PAGE_COLUMNS = ["route", "sync_tokens", "latency_s", "model_cost_usd", "quality_score"]
PAGE_FORMATS = {
    f"{column}_{side}": fmt
    for column, fmt in (
        ("sync_tokens", "{:,.0f}"),
        ("latency_s", "{:.2f}"),
        ("model_cost_usd", "${:.6f}"),
        ("quality_score", "{:.2f}"),
    )
    for side in ("before", "after")
} | {"cost_saved_usd": "${:+.6f}"}
SAMPLE_PAGE = next(page for page in EVAL_SET if page.labels.privileged)
LOW_PAGE = next(
    page for page in EVAL_SET if page.labels.risk == "low" and not page.labels.privileged
)
HIGH_PAGE = next(page for page in EVAL_SET if page.labels.risk == "high")


def guardrail_failures(variant):
    return Counter(
        code
        for trace in TRACES[variant]
        for code in deterministic_guardrail_check(PAGE_BY_ID[trace.doc_id], trace)
    )


def symptoms(variant):
    row = SUMMARY_BY_VARIANT.loc[variant]
    models = frame.loc[frame["variant"] == variant, "models_used"].value_counts()
    print(
        f"{VARIANT_LABELS[variant]}: {row['mean_tool_calls']:.1f} tool calls per page "
        f"({row['mean_extra_tool_calls']:.1f} unneeded), {row['mean_sync_tokens']:,.0f} sync tokens, "
        f"p50 {row['p50_latency_s']:.2f}s, model cost ${row['model_cost_per_page_usd']:.5f} per page"
    )
    print(f"  escalation rate {row['escalation_rate']:.0%}; privileged leaks {row['privileged_leaks']}, PII leaks {row['pii_leaks']}")
    print("  guardrail failures:", dict(guardrail_failures(variant)) or "none")
    print("  models used:", {name: int(count) for name, count in models.items()})


def request_facts(request):
    text = request.get("text", {})
    choice = request.get("tool_choice", "-")
    return {
        "model": request["model"],
        "reasoning": request.get("reasoning", {}).get("effort", "-"),
        "verbosity": text.get("verbosity", "-"),
        "structured_output": text.get("format", {}).get("name", "free text"),
        "max_output_tokens": request.get("max_output_tokens", "-"),
        "tool_choice": choice if isinstance(choice, str) else json.dumps(choice)[:40],
        "tools": ", ".join(sorted(tool["name"] for tool in request.get("tools", []))) or "-",
        "questions": ", ".join(q["name"] for q in request.get("questions", [])) or "-",
        "prompt_cache_key": request.get("prompt_cache_key", "-"),
        "instructions_chars": len(request.get("instructions", "")),
        "input_chars": len(json.dumps(request["input"])),
    }


def show_diff(before, after, page):
    old = request_facts(variant_request_example(before, page))
    new = request_facts(variant_request_example(after, page))
    rows = [{"setting": key, "before": old[key], "after": new[key]} for key in old if old[key] != new[key]]
    if rows:
        display(pd.DataFrame(rows).set_index("setting"))
    else:
        print("No request difference for this page.")


def short_json(value, limit=160):
    if isinstance(value, str):
        return value if len(value) <= limit else f"{value[:limit]}... [{len(value)} chars]"
    if isinstance(value, dict):
        return {key: short_json(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [short_json(item, limit) for item in value]
    return value


def compare_rounds(previous, current, rows=8):
    before = per_page_view(frame, [previous], PAGE_COLUMNS).set_index("doc_id")
    after = per_page_view(frame, [current], PAGE_COLUMNS).set_index("doc_id")
    merged = before.drop(columns="variant_label").add_suffix("_before").join(
        after.drop(columns="variant_label").add_suffix("_after")
    )
    merged["cost_saved_usd"] = merged["model_cost_usd_before"] - merged["model_cost_usd_after"]
    changed = int((merged["route_before"] != merged["route_after"]).sum())
    print(f"Pages whose route changed: {changed} of {len(merged)}. Largest per-page cost changes:")
    display(merged.sort_values("cost_saved_usd", ascending=False).head(rows).style.format(PAGE_FORMATS))


def delta_row(variant):
    show(impact[summary["variant"] == variant])


def gate_result(current):
    row = SUMMARY_BY_VARIANT.loc[current]
    failed = sum(1 for trace in TRACES[current] if deterministic_guardrail_check(PAGE_BY_ID[trace.doc_id], trace))
    rate = 1 - failed / len(TRACES[current])
    hard_gates = row["privileged_leaks"] == 0 and row["pii_leaks"] == 0
    return rate, bool(hard_gates and rate >= GUARDRAIL_FLOOR)


def quality_gate(previous, current):
    row = SUMMARY_BY_VARIANT.loc[current]
    failures = guardrail_failures(current)
    rate, accepted = gate_result(current)
    print(f"Hard gates: privileged pages produced = {row['privileged_leaks']}, PII left in produced text = {row['pii_leaks']}")
    print(f"Guardrail pass rate: {rate:.0%} (floor {GUARDRAIL_FLOOR:.0%}); failures: {dict(failures) or 'none'}")
    print(f"Mean quality: {SUMMARY_BY_VARIANT.loc[previous, 'mean_quality']:.2f} -> {row['mean_quality']:.2f}")
    print(f"{VARIANT_LABELS[current]}: {'ACCEPTED' if accepted else 'REJECTED'}")
"""),
    ]


def cells() -> list[Cell]:
    return [*_opening(), *_helpers()]

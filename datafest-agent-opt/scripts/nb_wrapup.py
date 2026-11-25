"""Notebook cells: walkthroughs, the before and after summary, tuning order, conclusion."""

# ruff: noqa: E501
from __future__ import annotations

from doc_review_opt.fixtures import EVAL_SET
from scripts.cells import Cell, code, md

PAGES = len(EVAL_SET)


def _walkthroughs() -> list[Cell]:
    return [
        md("""
## Before and after walkthroughs

Three pages, each with the baseline and the final workflow: personal data left in produced text, a privileged trap page, and a borderline page.
"""),
        code("""
WALK_COLUMNS = [
    "variant_label",
    "models_used",
    "route",
    "route_reason",
    "decision",
    "escalated",
    "tool_calls",
    "sync_tokens",
    "latency_s",
    "model_cost_usd",
    "pii_leak",
    "privileged_leak",
    "quality_score",
]
baseline_rows = frame[frame["variant"] == BASELINE]
leak_page = PAGE_BY_ID[baseline_rows[baseline_rows["pii_leak"]]["doc_id"].iloc[0]]
trap_page = next(
    page
    for page in EVAL_SET
    if page.labels.difficulty == "trap" and page.labels.privileged
)
borderline_page = next(
    page for page in EVAL_SET if page.labels.difficulty == "borderline"
)


def walkthrough(page, title):
    print(f"=== {title}: {page.doc_id} ({page.labels.difficulty}, custodian {page.custodian})")
    print(page.text[:420])
    print(
        f"Labels: route={page.labels.expected_route}, privileged={page.labels.privileged}, "
        f"contains_pii={page.labels.contains_pii}, must_escalate={page.labels.must_escalate}"
    )
    rows = frame[(frame["doc_id"] == page.doc_id) & frame["variant"].isin([BASELINE, SPLIT])]
    display(
        rows[WALK_COLUMNS]
        .reset_index(drop=True)
        .style.format({"sync_tokens": "{:,.0f}", "latency_s": "{:.2f}", "model_cost_usd": "${:.6f}", "quality_score": "{:.2f}"})
    )


walkthrough(leak_page, "Personal data left in produced text by the baseline")
"""),
        code("""
walkthrough(trap_page, "Privileged trap page")
"""),
        code("""
walkthrough(borderline_page, "Borderline page")
"""),
        md(f"""
## Before and after summary

The baseline against the final workflow (Round 5), same {PAGES} pages.
"""),
        code("""
base = SUMMARY_BY_VARIANT.loc[BASELINE]
final_row = SUMMARY_BY_VARIANT.loc[SPLIT]
SUMMARY_ROWS = [
    ("Mean quality", "mean_quality", "{:.2f}"),
    ("Route accuracy", "route_accuracy", "{:.0%}"),
    ("Privileged pages produced", "privileged_leaks", "{:d}"),
    ("PII left in produced text", "pii_leaks", "{:d}"),
    ("Escalation rate", "escalation_rate", "{:.0%}"),
    ("Mean tool calls per page", "mean_tool_calls", "{:.1f}"),
    ("Mean sync tokens per page", "mean_sync_tokens", "{:,.0f}"),
    ("p50 latency (s)", "p50_latency_s", "{:.2f}"),
    ("Model cost per page", "model_cost_per_page_usd", "${:.5f}"),
    ("Cost per page including attorney time", "cost_per_page_usd", "${:.3f}"),
    ("Blended cost per verified page", "blended_cost_per_verified_page", "${:.3f}"),
    ("Model cost at 100k pages", "monthly_model_cost_at_100k_pages", "${:,.0f}"),
]
display(
    pd.DataFrame(
        [
            {"metric": name, "baseline": fmt.format(base[col]), "final": fmt.format(final_row[col])}
            for name, col, fmt in SUMMARY_ROWS
        ]
    ).set_index("metric")
)
"""),
        md("""
## Recommended tuning order

1. **Measure first.** A labelled eval set, hard gates, and a blended cost that includes attorney time.
2. **Cut waste before changing models.** Controls (Round 1) are the cheapest, safest wins.
3. **Ask per step whether it needs generated text.** For a typed decision, try a decision model plus code router first (Round 3) and calibrate it. It wins on latency and auditability, and on price against a large LLM but not necessarily the smallest (live: `gpt-6-luna` was cheaper). For generated or extracted text, use the smallest LLM that holds quality (Round 2). Here routing comes first to show the progression; start with this question today.
4. **Add a stronger model only where it earns its cost.** A second opinion on the ambiguous band beats sending everything to the top tier or an attorney.
5. **Cache only at volume.** Stable protocol first, a key per matter, and check cached tokens live (Round 4). On 6 pages caching cost more ($0.00021 against $0.00008 per page).
6. **Split work that can wait.** Batch the privilege log, QA and tags (Round 5), and report synchronous and background cost separately.
7. **Keep the human gate and the monitors.** ID-and-score-only escalation, deterministic guardrails, and a judge on sampled live traces.
8. **Re-run the loop** when a model, price, protocol or matter changes, and recalibrate when the engine changes.
"""),
        md("""
## Conclusion
"""),
        code("""
cost_cut = 1 - final_row["model_cost_per_page_usd"] / base["model_cost_per_page_usd"]
latency_cut = 1 - final_row["p50_latency_s"] / base["p50_latency_s"]
token_cut = 1 - final_row["mean_sync_tokens"] / base["mean_sync_tokens"]
engines = engine_comparison_table(engine_runs_calibrated).set_index("engine")
ratio = engines.loc["llm-terra", "cost_per_page_usd"] / engines.loc["decisions", "cost_per_page_usd"]
display(
    Markdown(
        f"On this synthetic eval set the sprint cut model cost per page by {cost_cut:.0%}, "
        f"synchronous tokens by {token_cut:.0%} and p50 latency by {latency_cut:.0%}, while "
        f"privileged pages produced were {base['privileged_leaks']} before and {final_row['privileged_leaks']} after, "
        f"and personal data left in produced text went from {base['pii_leaks']} to {final_row['pii_leaks']}. "
        f"Asked the same questions, `gpt-6.1-terra` costs about {ratio:.0f}x the Decisions call per page "
        f"(simulated, calibrated thresholds). These are simulated numbers, not a benchmark. The live run (2026-11-18, 6 pages) is in the section What the live run showed."
    )
)
"""),
        md("""
What to take away:

- **Measure, change one lever, gate on quality.** Each round was accepted or rejected on the gates before its savings counted.
- **A decision model wins on typed decisions.** Live, Decisions was about 6x faster than `gpt-6-luna` and 6 to 8x faster than `gpt-6.1-terra`, and 4 to 6x cheaper than `gpt-6.1-terra`. Its probabilities can be calibrated, audited and routed in code. Against the smallest LLM it was 1.7x pricier, so that case rests on latency and auditability.
- **It does not replace an LLM.** It cannot generate text, extract to your schema, call tools or explain itself, it runs only on `gpt-6-luna`, and it can refuse. Keep an LLM for redaction text and second opinions, and fail safe to an attorney.
- **Calibrate per engine.** A score is an expected value, confidence is not correctness, and confidence does not port between engines.
- **Live agreed with the simulation on direction, not every detail.** Controls and the decision model cut tokens, latency and cost; routing was slower and scored lower; caching cost more on 6 pages. Trust measured numbers.
- **Blend in the human cost.** The cheapest model per page is not the cheapest workflow if more pages reach an attorney.

**Caveats.** Pages are synthetic. Simulated token, latency, cache and accuracy numbers are illustrative formulas, the attorney cost per page is an assumption, and tool selection uses fixture labels. The live run (2026-11-18, 6 pages) is an anecdote. Use larger live traces and attorney-graded evaluations before deploying.
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
    return _walkthroughs()

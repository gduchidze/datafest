"""Notebook cells: the live tool loop, optimization rounds 1 to 5 and the attorney step."""

# ruff: noqa: E501
from __future__ import annotations

from doc_review_opt.fixtures import EVAL_SET
from scripts.cells import Cell, code, md

PAGES = len(EVAL_SET)


def _live_loop() -> list[Cell]:
    return [
        md("""
## Optional: live tool loop

The live helper in [tool_agent.py](../src/doc_review_opt/agent/classifiers/tool_agent.py) keeps reasoning and tool-call items between calls. After `max_tool_rounds` it sets `tool_choice="none"`. An incomplete or unparseable reply counts as a parse failure and the page goes to an attorney.

The example takes allowed tools from fixture labels; use evaluated logic in production. Live runs use the same graph and checks as the simulation, and every call is charged to the $1 budget.
"""),
        code("""
from doc_review_opt.agent.classifiers.tool_agent import live_config_for_page
from doc_review_opt.exceptions import BudgetExceededError
from doc_review_opt.experiment.live import pick_live_pages, run_live_variant

LIVE_PAGES = pick_live_pages(EVAL_SET)

live_page = SAMPLE_PAGE
print(live_config_for_page(live_page, CONTROLS))
"""),
        code("""
if RUN_LIVE_API_CALLS:
    live_run = await run_live_variant(
        CONTROLS, [live_page], client=client, budget=budget, limit=1
    )
    live_trace = live_run.traces[0]
    display(
        pd.DataFrame(
            [
                {
                    "doc_id": live_trace.doc_id,
                    "route": live_trace.route,
                    "tool_calls": live_trace.tool_calls,
                    "latency_s": live_trace.latency_s,
                    "model_cost_usd": live_trace.model_cost_usd,
                    "input_tokens": live_trace.input_tokens,
                    "output_tokens": live_trace.output_tokens,
                    "trace_url": live_run.trace_url,
                }
            ]
        )
    )
    print(f"Running spend: ${budget.total_usd:.4f} of ${budget.limit_usd:.2f}")
else:
    print("Dry-run mode. Set OPENAI_API_KEY and RUN_LIVE_API_CALLS=true to run a live Responses API review page.")
"""),
    ]


def _round_one() -> list[Cell]:
    return [
        md(
            """
## Round 1: prompt, tool and context controls

What to look at: left is the baseline, right is the change: fewer tools and a strict answer.
"""
        ),
        code("""
draw_diagram("round1")
"""),
        md("""
**Symptom.** The baseline calls every tool on every page, carries large payloads, and returns free text that sometimes cannot be parsed. It fails the personal-data gate.
"""),
        code("""
symptoms(BASELINE)
quality_gate(BASELINE, BASELINE)
"""),
        md("""
**Lever.** Same model, tighter rules:

- a controlled prompt with the fewest-tools and never-produce-privileged rules
- slim schemas and payloads
- `allowed_tools` per page type
- `verbosity="low"`, a strict `json_schema` answer and a `max_output_tokens` cap

Expected: far fewer tokens and no parse failures.

**Diff.** One privileged page, before and after.
"""),
        code("""
show_diff(BASELINE, CONTROLS, SAMPLE_PAGE)
"""),
        code("""
from doc_review_opt.agent.prompts import BAD_BASELINE_PROMPT, CONTROLLED_PROMPT

print("--- baseline prompt ---")
print(BAD_BASELINE_PROMPT)
print("--- controlled prompt ---")
print(CONTROLLED_PROMPT)
"""),
        md(f"""
**Re-measure.** Same {PAGES} pages, before and after.
"""),
        code("""
compare_rounds(BASELINE, CONTROLS)
delta_row(CONTROLS)
"""),
        md("""
**Quality gate.** Accepted only if both hard gates are clear and the guardrail pass rate stays above the floor in `config.toml`.
"""),
        code("""
quality_gate(BASELINE, CONTROLS)
"""),
        md("""
**Tradeoff.** The schema and allow-list are logic you now maintain, and the allow-list uses fixture labels. The strongest model still handles every page, which Round 2 addresses.
"""),
        code("""
display(plot_progress(summary, upto_variant=CONTROLS))
"""),
    ]


def _round_two() -> list[Cell]:
    return [
        md(
            """
## Round 2: model selection

What to look at: cheap luna for routine pages, terra only for risky ones.
"""
        ),
        code("""
draw_diagram("round2")
"""),
        md("""
**Symptom.** Round 1 spends `gpt-6.1-terra` tokens on obvious pages. Only privileged and borderline pages need the top tier.
"""),
        code("""
symptoms(CONTROLS)
"""),
        md("""
**Lever.** A cheap `gpt-6-luna` triage step (`responsive`, `privileged`, `pii`, `needs_attorney`). Routine pages stay on `gpt-6-luna`; risky ones go to `gpt-6.1-terra`.

**Diff.** A low-risk page runs on the small model; a high-risk page stays on the strong one.
"""),
        code("""
print("Low-risk page:")
show_diff(CONTROLS, ROUTING, LOW_PAGE)
print("High-risk page:")
show_diff(CONTROLS, ROUTING, HIGH_PAGE)
"""),
        code("""
from doc_review_opt.agent.classifiers.tool_agent import triage_request

print(json.dumps(short_json(triage_request(LOW_PAGE), 220), indent=2))
"""),
        md("""
**Re-measure.**
"""),
        code("""
compare_rounds(CONTROLS, ROUTING)
delta_row(ROUTING)
"""),
        md("""
**Quality gate.** Check the cheap model's mistakes in the guardrail failures, not only the hard gates.
"""),
        code("""
quality_gate(CONTROLS, ROUTING)
"""),
        md("""
**Tradeoff.** Triage adds tokens to every page, and a wrong triage becomes a wrong escalation. Triage is also an LLM doing classification, which opens the door to Round 3.

**Live (2026-11-18, 6 pages).** Round 2 was slower than Round 1 (p50 4.00s against 2.96s) and scored lower (quality 0.95 against 0.98). Model cost per page did fall ($0.00039 against $0.00112). The simulated latency gain is a formula, not a measurement.
"""),
        code("""
display(plot_progress(summary, upto_variant=ROUTING))
"""),
    ]


def _round_three() -> list[Cell]:
    return [
        md(
            """
## Round 3: replace the LLM with a decision model

What to look at: who decides: a typed Decisions call and plain code, with an LLM only on the ambiguous band.
"""
        ),
        code("""
draw_diagram("round3")
"""),
        md("""
**Symptom.** The agent still uses an LLM to classify: responsive, privileged, personal data. Its confidence is not a probability and the thresholds are buried in a prompt. This is a typed decision, which the Decisions API answers directly.
"""),
        code("""
symptoms(ROUTING)
"""),
        md("""
**Lever.** One Decisions call on `gpt-6-luna` asks three typed questions: a `score` for responsiveness (three levels) and `predicate`s for privileged and contains-personal-data. **Plain code** routes the answers:

- privilege probability above the threshold goes to an attorney (privilege beats personal data)
- responsiveness below the not-responsive threshold is withheld
- an ambiguous score, or a responsive score with low confidence, goes to an attorney
- otherwise personal data above its threshold is redacted, else produced

An LLM stays only where Decisions cannot help: redaction text (`gpt-6-luna`) and a `gpt-6.1-terra` second opinion on the ambiguous band. A prior-production check in code skips the model call entirely. Thresholds are calibrated on the calibration split, never the holdout.

**Diff.** The Round 2 request next to the Decisions request. The protocol goes first in `input` so it can be cached; the policy lives in each question.
"""),
        code("""
from doc_review_opt.agent.classifiers.decisions import (
    DECISIONS_MODEL,
    build_decisions_request,
)
from doc_review_opt.agent.routing import (
    DEFAULT_POLICY,
    decide,
)
from doc_review_opt.fixtures import PROTOCOL

show_diff(ROUTING, DECISION_MODEL, SAMPLE_PAGE)
print(json.dumps(short_json(build_decisions_request(SAMPLE_PAGE, PROTOCOL), 200), indent=2))
"""),
        md("""
The typed answer for the same page (simulated): a `score` with level probabilities and a confidence, and one probability per predicate.
"""),
        code("""
from doc_review_opt.experiment.simulation import SimulatedTypedClassifier

sample = await SimulatedTypedClassifier("decisions").classify(SAMPLE_PAGE, PROTOCOL)
print(sample.model_dump_json(indent=2))
"""),
        code("""
if RUN_LIVE_API_CALLS:
    from doc_review_opt.agent.classifiers.decisions import DecisionsClassifier
    from doc_review_opt.experiment.live import BudgetedClassifier

    budget.check()
    live_sample = await BudgetedClassifier(DecisionsClassifier(client), budget).classify(
        SAMPLE_PAGE, PROTOCOL
    )
    print(live_sample.model_dump_json(indent=2))
    print(f"Running spend: ${budget.total_usd:.4f} of ${budget.limit_usd:.2f}")
else:
    print("Dry-run mode. Set OPENAI_API_KEY and RUN_LIVE_API_CALLS=true to send this page to the real Decisions API.")
"""),
        md("""
The router is ordinary code, so reading it is the audit. Each outcome has a named reason, a refusal or error can only escalate, and each threshold is one tunable number.
"""),
        code("""
print(inspect.getsource(decide))
print(inspect.getsource(type(DEFAULT_POLICY)))
"""),
        md("""
**Re-measure.** Round 2 against full Round 3, then Decisions alone against Decisions plus the second opinion.
"""),
        code("""
compare_rounds(ROUTING, DECISION_MODEL)
delta_row(DECISION_MODEL)
show(
    full_summary[full_summary["variant"].isin([ROUTER_ONLY, DECISION_MODEL])][
        [
            "variant_label",
            "mean_quality",
            "route_accuracy",
            "escalation_rate",
            "p50_latency_s",
            "model_cost_per_page_usd",
            "cost_per_page_usd",
        ]
    ]
)
"""),
        md("""
**Quality gate.** Decisions alone is fast and cheap, but its mistakes show in the guardrails. That is why the second opinion exists.
"""),
        code("""
quality_gate(ROUTING, DECISION_MODEL)
print()
quality_gate(ROUTING, ROUTER_ONLY)
"""),
        md("""
### Lever: put the protocol in the question

**The policy lives in the question.** The protocol says an attorney merely copied on business mail does not make a page privileged. The first `privileged` question only said "a named attorney is the author or recipient". In the simulation that over-flags cc'd mail. The fix is in the question text, not the thresholds. The two texts side by side:
"""),
        code("""
import difflib

from doc_review_opt.agent.classifiers.decisions import (
    ORIGINAL_PRIVILEGE_INSTRUCTIONS,
    PROTOCOL_PRIVILEGE_INSTRUCTIONS,
)

print("\\n".join(difflib.unified_diff(
    ORIGINAL_PRIVILEGE_INSTRUCTIONS.splitlines(),
    PROTOCOL_PRIVILEGE_INSTRUCTIONS.splitlines(),
    fromfile="original wording",
    tofile="protocol wording",
    lineterm="",
)))
"""),
        md("""
**Ablation.** `03b` is Round 3 with the original wording; `03_decision_model` uses the protocol wording. Read the verdicts from the table.

**Simulation assumption.** The dry-run privileged probability for cc'd mail under the protocol wording is a `config.toml` value, not a measurement.

**A separate live probe refuted it (2026-11-18).** The real API scored ARC-000306 at 0.02 (original wording) and 0.00 (protocol wording), and ARC-000413 at 0.00 with both. The simulated over-escalation of about 0.3 did not reproduce, so `03b` is a simulation-only illustration. The protocol wording stays because it is free and matches the protocol.
"""),
        code("""
import pandas as pd

ablation_rows = []
for variant in (ORIGINAL_WORDING_VARIANT, DECISION_MODEL):
    rate, accepted = gate_result(variant)
    row = SUMMARY_BY_VARIANT.loc[variant]
    ablation_rows.append(
        {
            "variant": VARIANT_LABELS[variant],
            "escalation_rate": row["escalation_rate"],
            "unnecessary_attorney_reviews": sum(t.unnecessary_escalation for t in TRACES[variant]),
            "privileged_leaks": row["privileged_leaks"],
            "guardrail_pass": rate,
            "quality_gate": "ACCEPTED" if accepted else "REJECTED",
        }
    )
display(
    pd.DataFrame(ablation_rows).style.format(
        {"escalation_rate": "{:.0%}", "guardrail_pass": "{:.0%}"}
    )
)
print()
quality_gate(ROUTING, ORIGINAL_WORDING_VARIANT)
"""),
        code("""
from doc_review_opt.agent.classifiers.decisions import ORIGINAL_WORDING
from doc_review_opt.experiment.simulation import simulated_answers

cc_pages = [page for page in EVAL_SET if page.cc and not page.labels.privileged]
if RUN_LIVE_API_CALLS:
    from doc_review_opt.agent.classifiers.decisions import DecisionsClassifier
    from doc_review_opt.experiment.live import BudgetedClassifier

    budget.check(reserve_usd=2 * len(cc_pages) * LIVE.per_page_ceiling_usd)
    for page in cc_pages:
        for label, wording in (("original", ORIGINAL_WORDING), ("protocol", "protocol")):
            live = await BudgetedClassifier(
                DecisionsClassifier(client, wording=wording), budget
            ).classify(page, PROTOCOL)
            answer = live.answers.get("privileged")
            print(f"LIVE {page.doc_id} {label:>8} wording: privileged probability = {getattr(answer, 'probability', None)}")
    print(f"Running spend: ${budget.total_usd:.4f} of ${budget.limit_usd:.2f}")
else:
    for page in cc_pages:
        for label, wording in (("original", ORIGINAL_WORDING), ("protocol", "protocol")):
            probability = simulated_answers(page, "decisions", wording)["privileged"].probability
            print(f"SIMULATION ASSUMPTION {page.doc_id} {label:>8} wording: privileged probability = {probability:.3f}")
    print("Dry-run mode: set RUN_LIVE_API_CALLS=true to measure the real Decisions probabilities on these pages.")
"""),
        md("""
The escalation came from one question's wording, so in the simulation the fix is one sentence, already in the protocol. Live, the lever was not needed (see "What the live run showed").
"""),
        md("""
### Uncalibrated vs calibrated thresholds

The default privilege threshold of 0.2 was tuned for another engine. `calibrate_policy` searches a small grid (privilege, PII, responsiveness confidence) on the **calibration split only**. It picks the cheapest policy with zero hard-gate leaks, breaking ties by distance from the default. The holdout then shows whether it generalizes.

Under the protocol wording, cc'd business mail already scores below 0.2 for privilege, so a higher bar has little to gain. The printout shows the picked threshold next to the default. In production, recalibrate whenever the engine, prompt or matter changes.
"""),
        code("""
from doc_review_opt.experiment.tradeoffs import (
    calibrate_policy,
    engine_comparison_table,
    policy_report,
    simulate_engine_comparison,
)
from doc_review_opt.fixtures import CALIBRATION, HOLDOUT

engine_runs_default = await simulate_engine_comparison()
engine_runs_calibrated = await simulate_engine_comparison(policy=CALIBRATED_POLICY)
decisions_results = {
    result.doc_id: result.classification
    for result in engine_runs_default["decisions"].outcome.results
}
picked = calibrate_policy(decisions_results, CALIBRATION)
print("Picked on the calibration split:", picked)
print(
    "Privilege escalation threshold: default",
    DEFAULT_POLICY.privilege_escalation,
    "-> calibrated",
    picked.privilege_escalation,
)
print("Matches the policy stored in config.toml:", picked == CALIBRATED_POLICY)
policy_table = policy_report(
    decisions_results,
    {"calibration": CALIBRATION, "holdout": HOLDOUT},
    {"default": DEFAULT_POLICY, "calibrated": CALIBRATED_POLICY},
)
display(
    policy_table.style.format(
        {
            "escalation_rate": "{:.0%}",
            "mean_quality": "{:.2f}",
            "blended_cost_per_page_usd": "${:.3f}",
        }
    )
)
"""),
        md("""
**How to read it.** Each pair is one split. Yellow is the default thresholds, green the calibrated ones.
"""),
        code("""
from doc_review_opt.experiment.tradeoffs import plot_calibration

display(plot_calibration(policy_table))
"""),
        md("""
The calibrated policy keeps the privilege threshold unchanged, since cc'd pages already score below 0.2. The simulated original-wording ablation (Round 3b) is where those pages score higher. The grid searches thresholds only, so the quality gate still decides and rejected rounds stay rejected.
"""),
        md("""
### Three lessons

**1. A score is an expected value, not a class.** A `score` question returns the probability-weighted average of its levels. A page split across two levels lands between them, say 1.4, and goes to the ambiguous band. Do not read 1.4 as "level 1".
"""),
        code("""
spread_page = next(
    page
    for page in EVAL_SET
    if page.labels.responsive_level == 2
    and decisions_results[page.doc_id].answers["responsive"].score < 1.5
)
answer = decisions_results[spread_page.doc_id].answers["responsive"]
print(f"{spread_page.doc_id}: labelled responsive level {spread_page.labels.responsive_level}")
print("level probabilities:", [round(p, 2) for p in answer.probabilities])
print("expected value     :", round(sum(i * p for i, p in enumerate(answer.probabilities)), 2))
print("reported score     :", round(answer.score, 2))
"""),
        md("""
**2. Confidence is not correctness.** It measures how sharp the distribution is, not whether the label is right. In the router it only gates "responsive" to "produce"; it never gates privilege, personal data or the not-responsive withhold. The table splits each engine's confidence by whether its rounded score matched the label.
"""),
        code("""
confidence_rows = [
    {
        "engine": name,
        "correct_level": round(result.classification.answers["responsive"].score)
        == PAGE_BY_ID[result.doc_id].labels.responsive_level,
        "confidence": result.classification.answers["responsive"].confidence,
    }
    for name in ("decisions", "llm-luna", "llm-terra")
    for result in engine_runs_default[name].outcome.results
]
confidence_table = pd.DataFrame(confidence_rows)
display(
    confidence_table.groupby(["engine", "correct_level"])["confidence"]
    .agg(["count", "mean", "max"])
    .style.format({"mean": "{:.2f}", "max": "{:.2f}"})
)
"""),
        md("""
**3. Confidence does not port across engines.** The same threshold means different things per engine. Below, one router and one set of thresholds are applied to Decisions and to two LLMs asked for the same probabilities. Escalation and accuracy move a lot, and recalibrating changes them again.
"""),
        code("""
ENGINE_FORMATS = {
    "accuracy_vs_labels": "{:.0%}",
    "agreement_vs_decisions": "{:.0%}",
    "cost_per_page_usd": "${:.5f}",
    "cost_per_100k_pages_usd": "${:,.0f}",
    "p50_latency_s": "{:.2f}",
    "mean_sync_tokens": "{:,.0f}",
    "escalation_rate": "{:.0%}",
}
print("Default thresholds:")
display(engine_comparison_table(engine_runs_default).drop(columns="label").style.format(ENGINE_FORMATS))
print("Calibrated thresholds:")
display(engine_comparison_table(engine_runs_calibrated).drop(columns="label").style.format(ENGINE_FORMATS))
"""),
        md("""
### What Decisions can't do

| Need | Why a decision model does not cover it | What to use |
|---|---|---|
| Generate text (redacted page, privilege-log entry) | Answers are typed numbers and choices only | `gpt-6-luna` Responses call, or a Batch job |
| Extract spans into your own schema | There is no free-form output | An LLM with a strict `json_schema` |
| Call tools or look things up | There is no tool loop | The code around the call, or an agent step |
| Explain itself for a privilege log | It returns probabilities, not reasons | An LLM, run after the decision and only where a reason is required |
| Choose a stronger model | Only `gpt-6-luna` is supported | `gpt-6.1-terra` as a second opinion on the ambiguous band |
| Always answer | Any answer can come back as a refusal | Fail safe: treat a refusal as "ask an attorney" |

A refusal never produces or withholds a page; it escalates.
"""),
        code("""
from doc_review_opt.models import ClassifierResult, RefusalAnswer

refused = ClassifierResult(
    engine="decisions",
    model=DECISIONS_MODEL,
    answers={"privileged": RefusalAnswer(name="privileged")},
)
print(decide(refused))
"""),
        md("""
**Tradeoff.** A decision model is far cheaper than `gpt-6.1-terra` and several times faster than either LLM, and its numbers can be calibrated and audited. It is not always cheaper than the smallest LLM.

**Live (2026-11-18, 6 pages).** Decisions was about 6x faster than `gpt-6-luna` as an LLM classifier but about 1.7x more expensive per page. Against `gpt-6.1-terra` it was 4 to 6x cheaper and 6 to 8x faster.

You also own the thresholds, cannot ask it to explain itself, and its edge mistakes are why a second opinion sits before the attorney. Compare model cost, escalation rate and blended cost per page with Round 2: an extra attorney review costs far more than the model calls it replaces.
"""),
        code("""
display(plot_progress(summary, upto_variant=DECISION_MODEL))
"""),
    ]


def _round_four_five() -> list[Cell]:
    return [
        md(
            """
## Round 4: prompt caching

What to look at: the order: the part that never changes comes first, so it can be cached.
"""
        ),
        code("""
draw_diagram("round4")
"""),
        md("""
**Symptom.** Every Decisions call resends the same protocol before the page, billed as ordinary input each time.
"""),
        code("""
symptoms(DECISION_MODEL)
"""),
        md("""
**Lever.** Protocol first in the Decisions `input`, page last, and a `prompt_cache_key` per matter.

- Only an `input` prefix of at least 1,024 tokens is cached.
- **Questions are never cached**, so keep them short.
- A cache write costs 1.25x the input price, so the prefix must be reused enough.

**Live vs simulated.** Live caching here is Decisions-only. Round 3 may already cache an implicit prefix, so live Round 3 against Round 4 does not isolate this lever. Redaction-prefix caching is simulation-only.

**Diff.**
"""),
        code("""
from doc_review_opt.agent.classifiers.decisions import QUESTIONS, protocol_block
from doc_review_opt.experiment.live import CACHE_KEY
from doc_review_opt.pricing import CACHE_MIN_TOKENS

print(f"Stable protocol prefix: about {approx_tokens(protocol_block(PROTOCOL)):,} tokens (minimum for caching: {CACHE_MIN_TOKENS:,})")
print(f"Questions re-billed on every call (never cached): about {approx_tokens(QUESTIONS):,} tokens")
print("prompt_cache_key:", CACHE_KEY)
show_diff(DECISION_MODEL, CACHING, SAMPLE_PAGE)
"""),
        md("""
**Re-measure.**
"""),
        code("""
compare_rounds(DECISION_MODEL, CACHING)
delta_row(CACHING)
display(
    frame[frame["variant"] == CACHING][
        ["doc_id", "input_tokens", "cached_tokens", "cache_write_tokens", "model_cost_usd"]
    ]
    .head(6)
    .style.format({"model_cost_usd": "${:.6f}"})
)
"""),
        md("""
**Quality gate.** Caching changes billing, not answers, so quality must be unchanged.
"""),
        code("""
quality_gate(DECISION_MODEL, CACHING)
"""),
        md("""
**Tradeoff.** The first request pays the 1.25x write; later ones read the prefix at a tenth of the input price. Caching needs a stable prefix and steady traffic. The simulation assumes a warm cache, so treat the saving as an upper bound.

**Live (2026-11-18, 6 pages).** Round 4 cost more than Round 3 ($0.00021 against $0.00008 per page). Cache writes only amortize over volume, such as a steady stream of pages for one matter.
"""),
        code("""
display(plot_progress(summary, upto_variant=CACHING))
"""),
        md(
            """
## Round 5: split the workflow

What to look at: two lanes: the review path stays fast, and the slow jobs run later in Batch.
"""
        ),
        code("""
draw_diagram("round5")
"""),
        md("""
**Symptom.** The privilege-log draft, QA sampling and production tags do not change the disposition, yet they sit in the review path.
"""),
        code("""
symptoms(CACHING)
"""),
        md("""
**Lever.** The review path stays classify, route, decide. The privilege-log draft, QA and tags become Batch jobs on `gpt-6-luna` at 50% off, run afterwards. Expected: lower p50 latency and synchronous cost. Total tokens stay level, and background cost is reported separately.

**Diff.** One Batch row per background task per page, written to `outputs/` and **not submitted**.
"""),
        code("""
from collections import Counter

from doc_review_opt.experiment.batch import build_background_batch, write_batch_jsonl

batch_rows = build_background_batch(EVAL_SET, TRACES[SPLIT])
batch_path = write_batch_jsonl(batch_rows, OUTPUT_DIR / "background_batch.jsonl")
print(f"Wrote {len(batch_rows)} Batch rows to {batch_path.name}")
print("Rows per task:", dict(Counter(row["custom_id"].split(":")[1] for row in batch_rows)))
print(json.dumps(short_json(batch_rows[0], 140), indent=2))
"""),
        md("""
Submission code is shown, never run. Batch has a 24-hour window and a 50% discount:

```python
batch_file = client.files.create(file=open(batch_path, "rb"), purpose="batch")
batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/responses",
    completion_window="24h",
)
print(batch.id, batch.status)
```

**Re-measure.**
"""),
        code("""
compare_rounds(CACHING, SPLIT)
delta_row(SPLIT)
show(
    summary[summary["variant"].isin([CACHING, SPLIT])][
        [
            "variant_label",
            "mean_sync_tokens",
            "mean_total_tokens",
            "p50_latency_s",
            "model_cost_per_page_usd",
        ]
    ].assign(
        sync_cost_per_page_usd=summary["sync_cost_per_page_usd"],
        background_cost_per_page_usd=summary["background_cost_per_page_usd"],
    )
)
"""),
        md("""
**Quality gate.** The split moves work out of the review path; it must not move any decision.
"""),
        code("""
quality_gate(CACHING, SPLIT)
"""),
        md("""
**Tradeoff.** Synchronous latency drops (compare p50 with Round 4) while total tokens stay level, at half price. It adds a job queue, result storage and failure handling, so split only work that can wait.
"""),
        code("""
display(plot_progress(summary, upto_variant=SPLIT))
"""),
    ]


def _scoreboard() -> list[Cell]:
    return [
        md("""
## Quality across rounds

Three views of the same gate: which versions pass, where the pages end up, and which pages each version gets wrong.
"""),
        md("""
**How to read it.** Each bar is one version. Green passes the gate (at least the floor and zero leaks); red is rejected. The dashed line is the floor.
"""),
        code("""
from doc_review_opt.reporting.reporting import plot_quality_gate

gate_rows = []
for variant in VARIANT_ORDER + EXTRA_VARIANTS:
    rate, accepted = gate_result(variant)
    gate_rows.append({"label": VARIANT_LABELS[variant], "pass_rate": rate, "accepted": accepted})
display(plot_quality_gate(pd.DataFrame(gate_rows), GUARDRAIL_FLOOR))
"""),
        md("""
**How to read it.** Each bar is 100% of the pages. The top bar is what the labels expect; a good round matches it.
"""),
        code("""
from doc_review_opt.reporting.reporting import plot_route_mix

display(plot_route_mix(frame, VARIANT_ORDER))
"""),
        md("""
**How to read it.** One row per page, one column per version. Green dot: right route. Red cross: wrong route. Red triangle: a leak.
"""),
        code("""
from doc_review_opt.reporting.reporting import plot_scorecard

display(plot_scorecard(frame, VARIANT_ORDER))
"""),
    ]


def _human_loop() -> list[Cell]:
    return [
        md(
            """
## Human in the loop: attorney review

What to look at: the graph pauses, only ids and scores go out, and the decision comes back.
"""
        ),
        code("""
draw_diagram("human_in_the_loop")
"""),
        md("""
Every page routed to attorney review pauses the graph with `interrupt()` and resumes with the attorney's decision. The payload holds IDs, the reason and scores only, never page text, so a review queue can be logged without leaking privileged content. The next cells inspect the payloads, assert none contains page text, and resume with scripted decisions.
"""),
        code("""
from langgraph.types import Command

from doc_review_opt.experiment.runs import scripted_attorney_decisions

escalating = [page for page in EVAL_SET if page.labels.must_escalate][:3]
clean = [page for page in EVAL_SET if not page.labels.must_escalate][:1]
hitl_pages = escalating + clean
hitl_classifier, hitl_redactor = build_simulated_engines(
    VARIANTS[DECISION_MODEL], hitl_pages, CALIBRATED_POLICY
)
hitl_graph = build_review_graph(
    hitl_classifier, hitl_redactor, review_checkpointer(), policy=CALIBRATED_POLICY
)
hitl_config = {"configurable": {"thread_id": "attorney-demo"}}
paused = await hitl_graph.ainvoke({"pages": hitl_pages, "results": []}, hitl_config)
interrupts = paused["__interrupt__"]
payloads = [pending.value for pending in interrupts]
print(f"{len(hitl_pages)} pages in, {len(paused['results'])} finished, {len(payloads)} waiting for an attorney")
display(pd.DataFrame(payloads))
"""),
        code("""
SNIPPET_CHARS = get_settings().quality.snippet_chars
for payload in payloads:
    page = PAGE_BY_ID[payload["doc_id"]]
    assert page.text[:SNIPPET_CHARS] not in json.dumps(payload)
    assert set(payload) == {"doc_id", "reason", "privileged", "responsive_score", "responsive_confidence"}
print("Checked: no interrupt payload contains page text.")
"""),
        code("""
attorney_decisions = scripted_attorney_decisions(hitl_pages)
resume = {pending.id: attorney_decisions[pending.value["doc_id"]] for pending in interrupts}
final = await hitl_graph.ainvoke(Command(resume=resume), hitl_config)
display(
    pd.DataFrame(
        [
            result.model_dump(
                include={"doc_id", "route", "decision", "escalated", "attorney_decision", "redacted", "reason"}
            )
            for result in final["results"]
        ]
    )
)
"""),
    ]


def cells() -> list[Cell]:
    return [
        *_live_loop(),
        *_round_one(),
        *_round_two(),
        *_round_three(),
        *_round_four_five(),
        *_scoreboard(),
        *_human_loop(),
    ]

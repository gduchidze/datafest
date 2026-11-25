# Optimizing document review agents for cost and quality: decision models vs LLMs

A demo notebook that optimizes a page-level e-discovery review agent (produce, withhold, redact, or send to an attorney) one lever at a time, and asks a question the source material does not: **which steps should be a decision model (the OpenAI Decisions API) and which still need an LLM?**

Only `gpt-6.1-terra`, `gpt-6-luna` and the Decisions API (`gpt-6-luna`) are used. Reasoning is off for every Responses call.

## How the agent works

Each page goes through one LangGraph subgraph (pages fan out with `Send()`):

1. **Classify once.** One Decisions call asks three typed questions: `responsive` (a 3-level `score`), `privileged` and `contains_pii` (`predicate` probabilities). The review protocol goes first in `input` so it can be cached; each question's instructions carry the policy.
2. **Route in code** (thresholds in `config.toml`):
   - privilege above 0.2 → attorney;
   - responsiveness below 0.5 → withhold;
   - the ambiguous band → a `gpt-6.1-terra` second opinion, then an attorney if still unclear;
   - PII above 0.5 → redact;
   - otherwise produce.
3. **Redact** with `gpt-6-luna` only when routed there.
4. **Attorney review** with `interrupt()`. The payload carries ids and scores, never page text, because it is stored in the checkpoint.

Safety rules:
- **Hard gates.** Zero privileged pages produced and zero PII left in produced text.
- **PII fails closed.** An unclear PII signal (error, refusal, timeout, unparsed reply) always means redact before production.
- **Max-merge.** A second opinion can raise a privilege or PII flag but never clear one.
- **Fail safe.** Any refusal or error sends the page to an attorney.

## Eval set

- **Pages.** 60 synthetic pages for one invented matter, *24-CV-118 (Hollis v. Arcadia Freight)*. This is the gist's 6 pages plus 54 new ones. The first 30 pages (the gist's 6 and 24 more) are unchanged; 30 harder pages were appended after them.
- **Split.** A calibration half of 30 pages, used to fit thresholds, and a holdout of 30, used to check them.
- **Expected routes.** 18 produce, 12 withhold, 20 attorney review, 10 redact.
- **Difficulty.** 30 simple, 13 traps, 10 borderline, 7 routine.
- **Traps.**
  - An attorney only on cc of business mail, which must still be produced.
  - Restatement talk with no attorney, which is not privileged.
  - A duplicate of a page already produced.
  - PII on a page that isn't responsive, which is withheld, not redacted.
  - Partial OCR scans that should go to an attorney.
  - Legal advice passed on second hand, or with no lawyer named.
  - A privilege stamp on a page with no lawyer and no advice, which is produced.
  - A long page with personal data buried at the end.

Quality is measured four ways:
- **Hard gates.** No privileged page produced, no PII left in produced text.
- **Quality score.** A 0–0.99 score with named penalties.
- **Deterministic guardrails.** Each failure has a named code, for example `privileged_page_produced`, `pii_in_produced_text` or `escalation_mismatch`.
- **LLM judge.** An optional, blind `gpt-6.1-terra` judge. A judge pass never overrides a deterministic failure.

A round is **accepted only if its guardrail pass rate is at least 95%**. Cost is reported as model cost and as blended cost, which adds attorney time at $1.50 per escalated page.

## The evals loop

The notebook is built around the "make evals a first-class citizen" loop: version 1 → traces → find errors → build evals → tune → climb.

| Loop step | Where it lives | Status |
|---|---|---|
| **Version 1** | `00_bad_baseline`, a deliberately wasteful `gpt-6.1-terra` super-agent, measured first | Done |
| **Traces** | One `Trace` row per page per round, with the same schema offline and live (`traces.py`). Live runs also go to LangSmith | Done |
| **Find errors** | `deterministic_guardrail_check` failure codes, judge disagreements, and each round's "Symptom" step, which reads the previous round's traces | Done |
| **Build evals** | The 60-page eval set, guardrails and judge exist, but they were designed up front. Errors found in traces (for example the live Round 2 quality drop) are not yet turned into new regression cases | **Partial** |
| **Tune** | One lever per round, threshold calibration on the calibration split checked on holdout, and the privilege-question wording lever | Done |
| **Climb** | `round_impact` deltas, the "progress so far" chart after every round, and the quality gate deciding ACCEPTED or REJECTED | Done |

The open gap is the error-to-eval step: failing traces should become tagged regression cases that later rounds are graded on.

## Rounds

| Round | Lever | What changes |
|---|---|---|
| `00_bad_baseline` | none | `gpt-6.1-terra` super-agent, every tool on every page, verbose payloads, free-text answer, background work done synchronously |
| `01_prompt_tool_context_controls` | Prompt, tools, context | Tight prompt, `allowed_tools`, slim payloads, strict JSON disposition |
| `02_model_routing` | Model selection | `gpt-6-luna` triage, then routine pages on `gpt-6-luna` and risky pages on `gpt-6.1-terra` |
| `03_decision_model` | **Decisions vs LLM** | One Decisions call plus the code router, a `gpt-6.1-terra` second opinion on the ambiguous band, calibrated thresholds, and the protocol's cc rule in the privilege question |
| `04_prompt_caching` | Prompt caching | The ~1,250-token protocol prefix leads `input`, with a `prompt_cache_key` per matter |
| `05_split_workflow` | Sync/async split | Privilege log, QA and tags move to Batch at 50% off |
| `03a_router_only`, `03b_gist_wording` | Ablations | Decisions without the second opinion, and Round 3 with the gist's original privilege wording |

Every round follows the same six steps: symptom, lever, diff, re-measure, quality gate, tradeoff.

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group notebook
```

## Run offline (default, free)

Nothing calls an API unless you opt in. The simulation is deterministic and the whole notebook runs top to bottom in a few seconds.

```bash
uv run jupyter lab notebooks/optimizing_document_review_agents_for_cost_and_quality.ipynb
```

or headless, saving the outputs into the notebook:

```bash
RUN_LIVE_API_CALLS=false RUN_LLM_JUDGE=false LANGSMITH_TRACING=false \
  uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/optimizing_document_review_agents_for_cost_and_quality.ipynb
```

### Demo notebook (20 min)

A condensed notebook for a live talk, written in plain English. It opens with the system in one picture (engine, router, attorney), the LangGraph graph drawn from the compiled graph, and one four-page run that pauses for an attorney. Then it states what we optimize and the five levers, follows one privileged and one clean page through every round, and shows harder pages from the 60-page set. The rounds follow, one lever each, with LLMs vs Decisions right after Round 3.

```bash
uv run jupyter lab notebooks/optimizing_document_review_agents_demo.ipynb
RUN_LIVE_API_CALLS=false RUN_LLM_JUDGE=false LANGSMITH_TRACING=false \
  uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/optimizing_document_review_agents_demo.ipynb
```

## Run live

Copy `.env.example` to `.env` (git-ignored) and fill in the keys. Values already set in the environment win over `.env`.

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required for any live cell |
| `RUN_LIVE_API_CALLS=true` | Live tool loop, Decisions sample, engine comparison and the full live run |
| `RUN_LLM_JUDGE=true` | Optional answer judge (`gpt-6.1-terra`) over a few simulated traces |
| `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` | Send one LangSmith trace per variant and per engine; the notebook prints the URLs. `LANGSMITH_PROJECT` defaults to `doc-review-opt` |

All live cells share one `BudgetTracker` with a **$1 hard stop** (set in `src/doc_review_opt/config.toml`) and run on `LIVE_PAGE_LIMIT` pages (default 6). Before a live run the tracker reserves the per-page ceiling (classification plus redaction, from `config.toml`) for every page in the run, and refuses to start if that could cross the limit; the agent tool loop enforces the per-page classification ceiling by closing early with `tool_choice="none"`, so in-flight work cannot push spend past the limit. Judge calls reserve `max_concurrency` times the judge-call ceiling. A live demo takes a couple of minutes. The Batch example only writes `outputs/background_batch.jsonl`; its submission code is shown, never executed.

## Repository layout

The package has three layers. The agent is what reviews a document. The experiment layer measures and varies the agent. The reporting layer turns those measurements into charts, diagrams and tables.

The dependency rule: the agent imports neither the experiment layer nor the reporting layer, and never matplotlib, pandas or `langsmith.evaluation`. Experiments import the agent, and reporting sits on top of both. The shared foundation (`models`, `config`, `exceptions`, `pricing`, `fixtures`) imports from none of the three layers. `tests/test_architecture.py` enforces this by parsing the imports.

| Path under `src/doc_review_opt/` | Role |
|---|---|
| `config.toml`, `config.py` | Every threshold, price, model id and constant, loaded into frozen typed settings |
| `models.py`, `exceptions.py` | Frozen pydantic value types and the error types |
| `pricing.py` | Price table and cost estimator (cached input, cache writes, Batch discount) |
| `fixtures.py` | The matter, custodians, review protocol, prior production index and the 60 labelled pages with the calibration/holdout split |
| `agent/graph.py`, `state.py`, `nodes.py` | The LangGraph review graph, its state and interrupt payload, its nodes, the batch runner and `review_checkpointer()` |
| `agent/routing.py` | `RoutingPolicy` and the code router that turns typed answers into a route |
| `agent/protocols.py`, `redactor.py` | The `Classifier` and `Redactor` protocols and the OpenAI redactor |
| `agent/prompts.py`, `tools.py` | Prompt text, the mock review tools and their function schemas |
| `agent/variants.py` | The six optimization rounds as prompt, tool and config bundles |
| `agent/classifiers/` | The Decisions classifier, the LLM probability classifier, the tool-calling agent, the second-opinion cascade with the prior-production gate, and disposition and response parsing |
| `experiment/simulation.py`, `runs.py`, `traces.py` | Deterministic simulated engines, running a variant through the graph, trace schema, quality scoring and guardrails |
| `experiment/live.py`, `batch.py`, `evaluation.py` | Budget tracker, LangSmith tracing, live runs and the LLM judge, Batch JSONL, deterministic guardrails |
| `experiment/tradeoffs.py`, `scenarios.py` | Step matrix, engine comparison, threshold calibration and sweep, cascade analysis, architecture options and the cookbook fit score |
| `reporting/reporting.py`, `chart_style.py` | Summaries, round deltas and charts |
| `reporting/diagrams.py`, `walkthrough.py` | Diagrams drawn from `diagrams.toml` and the demo walkthrough tables |

The `notebooks/` directory contains both executed notebooks. The offline tests in `tests/` run both files headlessly.

## Simulated before and after

Generated by the offline simulation on the 60 synthetic pages (attorney time at $1.50 per escalated page):

| Variant | Quality | Privileged produced | PII left | Escalation | Sync tokens | p50 latency | Model cost / page | Cost / page with attorney |
|---|---|---|---|---|---|---|---|---|
| Bad baseline | 0.80 | 0 | 2 | 47% | 8,474 | 3.59s | $0.00895 | $0.709 |
| Round 1: controls | 0.98 | 0 | 0 | 33% | 1,335 | 1.59s | $0.00136 | $0.501 |
| Round 2: routing | 0.97 | 0 | 0 | 32% | 1,545 | 1.38s | $0.00049 | $0.475 |
| Round 3: decision model | 0.98 | 0 | 0 | 33% | 2,523 | 0.68s | $0.00072 | $0.501 |
| Round 4: caching | 0.98 | 0 | 0 | 33% | 2,747 | 0.65s | $0.00062 | $0.501 |
| Round 5: split workflow | 0.98 | 0 | 0 | 33% | 2,368 | 0.11s | $0.00062 | $0.501 |

Round 2 is cheapest per page but misses one escalation the labels require, so the notebook shows why cheaper is not always better. Round 3 cuts latency sharply. With the gist's `privileged` question wording, the second opinion cannot clear a privilege flag, so four attorney-cc'd business emails escalate unnecessarily and the round is rejected at 93% guardrail pass against the 95% floor (`03b_gist_wording`). Putting the review protocol's cc rule into the question ("policy lives in the question") removes those escalations, and the quality gate then accepts Rounds 1 to 5. That effect is a **simulation assumption** (`cc_privilege_with_protocol_wording` in `config.toml`); none of the 6 pages in the live run was attorney-cc'd, so the round results cannot show the wording effect; a separate recorded probe of the two original attorney-cc'd pages is below. The ablation, the gist-wording variant and the code-router-only variant are printed in the notebook rather than in this table. Asked the same questions, `gpt-6.1-terra` costs about 11x the Decisions call per page in the simulation. That probe did not reproduce the wording effect (see below).

## Live run (2026-11-18)

Live run, 2026-11-18, `LIVE_PAGE_LIMIT` = 6 pages chosen from the original 30-page set, one per route in turn, $0.18 of the $1 cap, no errors. The simulation above now uses 60 pages. These are measured, not modeled; your run will differ, and six pages are an anecdote, not a benchmark.

| Round | Quality | Escalation | Sync tokens | p50 (s) | Model $/page |
|---|---|---|---|---|---|
| Bad baseline | 0.84 | 33% | 16,996 | 10.92 | 0.02243 |
| R1 controls | 0.98 | 17% | 942 | 2.96 | 0.00112 |
| R2 routing | 0.95 | 17% | 1,326 | 4.00 | 0.00039 |
| R3 decision model | 0.98 | 17% | 1,772 | 0.45 | 0.00008 |
| R4 caching | 0.98 | 17% | 1,772 | 0.43 | 0.00021 |
| R5 split | 0.98 | 17% | 1,772 | 0.34 | 0.00008 |

Zero privileged or PII leaks in every round. Judge (`gpt-6.1-terra`): the baseline passed the judge on 83% of pages but passed the deterministic checks on 0%; Rounds 1 to 5 passed both on 100%.

Live engine comparison on the same 6 pages:

| Engine | Accuracy | Escalation | Model $/page | p50 (s) |
|---|---|---|---|---|
| Decisions | 100% | 17% | 0.00019 | 0.38 |
| `gpt-6-luna` as an LLM | 100% | 17% | 0.00011 | 2.26 |
| `gpt-6.1-terra` as an LLM | 100% | 17% | 0.00073 | 3.03 |
| Controlled agent (`gpt-6.1-terra`) | 100% | 17% | 0.00118 | 2.33 |

Separate live probe (none of the 6 pages above was attorney-cc'd): Decisions privileged probability on the two original attorney-cc'd mails (gist wording / protocol wording): ARC-000306 0.02 / 0.00; ARC-000413 0.00 / 0.00.

Where the live run diverged from the simulation:

1. **Round 2 routing was slower, not faster.** The extra triage call made Round 2 slower than Round 1 (p50 4.00s against 2.96s) and scored lower (quality 0.95 against 0.98). Per-page model cost did fall ($0.00039 against $0.00112).
2. **Decisions beat the LLMs on latency, not always on price.** In the live engine comparison Decisions was the fastest engine and `gpt-6-luna` as an LLM the cheapest; Decisions was about 6x faster than `gpt-6-luna` as an LLM classifier but about 1.7x more expensive per page. Against `gpt-6.1-terra` it was 4 to 6x cheaper and 6 to 8x faster. Its other advantages, calibratable probabilities and an auditable router, are not price.
3. **The wording lever was not needed live.** The real Decisions API already scores attorney-cc'd mail near 0 for privilege with either wording, so the simulated gist-wording over-escalation (about 0.3) did not reproduce. The `03b` ablation is a simulation-only illustration. The protocol wording stays: it is free and matches the protocol.
4. **Round 4 caching cost more than Round 3** on 6 pages ($0.00021 against $0.00008 per page). Cache writes are billed at 1.25x and only amortize over volume; caching pays off when the same prefix is reused many times within the cache lifetime, not on a short demo run.

## Caveats

- The pages are **synthetic**; the results are not a production benchmark.
- Token, latency, cache and accuracy numbers come from **illustrative formulas** with labelled per-engine ambiguity profiles. Replace them with live traces (the live cells emit the same trace schema) before deciding anything.
- The **attorney cost per page ($1.50)** is an assumption in `config.toml`; blended cost depends on it.
- As in the cookbook, tool selection and the allow-list use **oracle fixture labels**, so the simulation does not measure a model's ability to pick tools.
- Everything outside the "Live run" section above is simulated and illustrative; the live numbers are one 6-page run.
- Prices were verified on 2026-11-18; update `config.toml` when they change.

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pyright && uv run pytest
```
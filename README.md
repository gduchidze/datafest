# Not every agent decision deserves an LLM

Giorgi Duchidze’s DataFest talk explores when to use a decision model, when to call an LLM, and how to measure the trade-offs in quality, latency, and cost.

| Resource | What it covers |
| --- | --- |
| [Presentation](index.html) | The 30-slide talk, including Impel’s production case on screening incoming customer messages. Download the HTML file and open it in a browser. |
| [Demo notebook](notebooks/optimizing_document_review_agents_demo.ipynb) | A 20-minute walkthrough of a separate document review agent. Follow a page from classification to release, redaction, withholding, or attorney review. |
| [Full experiment](notebooks/optimizing_document_review_agents_for_cost_and_quality.ipynb) | Compare model routing, decision thresholds, caching, and batch processing across six rounds. |

## Run the notebooks

With Python 3.13 and [uv](https://docs.astral.sh/uv/) installed, run these commands from the repository root. Both notebooks run offline by default.

```bash
uv sync --frozen --group notebook
uv run --group notebook jupyter lab notebooks/
```

The [agent implementation](src/doc_review_opt/agent/) includes the classifiers and routing policy. For optional live API runs, see [.env.example](.env.example).

## What the experiment found

On a six-page live sample, Decisions had the lowest median latency (0.38 s). `gpt-6-luna` as an LLM had the lowest model cost ($0.00011 per page). All four engines chose the expected route on all six pages.

<details>
<summary>Dataset, simulated results, and live sample limits</summary>

- **Pages.** 60 synthetic pages from a fictional legal matter, comprising the six original examples plus 54 new ones.
- **Split.** A calibration half of 30 pages and a holdout of 30 pages.
- **Expected routes.** 18 produce, 12 withhold, 20 attorney review, 10 redact.
- **Difficulty.** 30 simple, 13 traps, 10 borderline, 7 routine.

The simulation assumes $1.50 per page sent to an attorney. PII means personally identifiable information.

| Variant | Quality | Privileged produced | PII left | Escalation | Sync tokens | p50 latency | Model cost / page | Cost / page with attorney |
|---|---|---|---|---|---|---|---|---|
| Bad baseline | 0.80 | 0 | 2 | 47% | 8,474 | 3.59s | $0.00895 | $0.709 |
| Round 1: controls | 0.98 | 0 | 0 | 33% | 1,335 | 1.59s | $0.00136 | $0.501 |
| Round 2: routing | 0.97 | 0 | 0 | 32% | 1,545 | 1.38s | $0.00049 | $0.475 |
| Round 3: decision model | 0.98 | 0 | 0 | 33% | 2,523 | 0.68s | $0.00072 | $0.501 |
| Round 4: caching | 0.98 | 0 | 0 | 33% | 2,747 | 0.65s | $0.00062 | $0.501 |
| Round 5: split workflow | 0.98 | 0 | 0 | 33% | 2,368 | 0.11s | $0.00062 | $0.501 |

On the same simulated classification task, `gpt-6.1-terra` costs about 11x the Decisions call. One prompt variant escalated four attorney-cc'd business emails unnecessarily and was rejected at 93% guardrail pass, below the 95% threshold.

The November 18, 2026 live run used 6 pages chosen from the original 30-page set, one per route in turn. On this sample, Decisions was the fastest engine and `gpt-6-luna` as an LLM the cheapest. The sample provides limited evidence about privilege detection because none of the 6 pages in the live run was attorney-cc'd.

Separate live probe (none of the 6 pages above was attorney-cc'd) — both prompt variants assigned a privilege probability of 0.02 or less to each of two business emails that copied an attorney.

</details>

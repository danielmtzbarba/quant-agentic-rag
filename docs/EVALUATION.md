# Evaluation Strategy

The evaluation stack now has two layers:

1. narrow deterministic regression tests for grounding behavior
2. a broader golden set and release-gate pass/fail check

The deterministic regression suite lives in:

- [tests/test_grounding_e2e.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/tests/test_grounding_e2e.py:1)
- [tests/test_grounding_repair.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/tests/test_grounding_repair.py:1)

The broader offline evaluation asset lives in:

- [data/evaluation/golden_set.json](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/data/evaluation/golden_set.json:1)

The aggregation logic lives in:

- [src/stock_agent_rag/evaluation.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/src/stock_agent_rag/evaluation.py:1)

## Golden Set

The current golden set contains 24 ticker/question pairs across multiple sectors and market regimes.

Each case includes:

- `ticker`
- `question`
- `sector`
- `market_regime`
- `expected_document_types`
- `relevant_source_ids`
- `required_issues`
- `prohibited_claims`
- `verdict_band`
- `requires_contradiction_review`

This is intended to be a release gate, not a benchmark for leaderboard-style model comparison.
The objective is to catch regressions in:

- grounding
- off-ticker retrieval leakage
- contradiction surfacing
- repair-loop recovery

## Metric Definitions

Release-gate evaluation currently tracks these metrics:

- `citation_format_compliance`
  Share of evaluated runs with zero malformed citations and zero prohibited placeholder strings.
- `unsupported_numeric_claim_rate`
  Total uncited numeric-claim violations divided by evaluated runs.
- `off_ticker_evidence_rate`
  Off-ticker retrieved evidence count divided by total retrieved evidence across evaluated runs.
- `contradiction_surfacing_rate`
  For golden-set cases marked `requires_contradiction_review=true`, the share of runs that surface at least one contradiction.
- `pass_rate_after_repair`
  Among runs that triggered the single repair loop, the share that ended in final verifier pass.
- `verification_pass_rate`
  Share of evaluated runs whose final verifier status is `pass`.
- `precision@k`
  Mean share of top-`k` retrieved source ids that are labeled relevant, computed only on cases with `relevant_source_ids`.
- `recall@k`
  Mean share of labeled relevant source ids recovered within top-`k`, computed only on cases with `relevant_source_ids`.
- `retrieval_label_coverage`
  Share of evaluated cases that actually have retrieval relevance labels.

## Release Gates

The current gate thresholds are defined in [src/stock_agent_rag/evaluation.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/src/stock_agent_rag/evaluation.py:1):

- full golden-set coverage required for the release run
- `citation_format_compliance >= 1.00`
- `unsupported_numeric_claim_rate <= 0.00`
- `off_ticker_evidence_rate <= 0.02`
- `contradiction_surfacing_rate >= 0.80`
- `pass_rate_after_repair >= 0.80`

These thresholds are intentionally strict on citation formatting and unsupported numeric claims. The workflow is now fail-closed on those issues, so the release gate should be aligned with that behavior.

## Runtime Metric Sources

The release-gate aggregator relies on metrics already emitted by the workflow and service layers:

- verifier metrics from [src/stock_agent_rag/workflow.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/src/stock_agent_rag/workflow.py:1121)
  - `malformed_citation_count`
  - `uncited_numeric_claim_count`
  - `prohibited_placeholder_count`
  - `repair_attempted`
- retrieval metrics from [src/stock_agent_rag/telemetry.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/src/stock_agent_rag/telemetry.py:78)
  - `off_ticker_evidence_count`
  - `off_ticker_evidence_rate`

## How To Run

The release-gate CLI expects a JSON file containing either:

- a top-level list of workflow result objects
- or an object with a `results` list

Command:

```bash
uv run stock-agent-rag release-gates --results path/to/results.json
```

Optional custom golden set:

```bash
uv run stock-agent-rag release-gates \
  --results path/to/results.json \
  --golden-set data/evaluation/golden_set.json \
  --retrieval-k 5

## Retrieval Labeling

`precision@k` and `recall@k` are now implemented in the evaluator, but they require
case-level `relevant_source_ids` labels in the golden set.

The current repo schema supports those labels. The remaining operational work is to annotate the
24 existing golden-set cases with the evidence ids that should appear near the top of retrieval for
each question.
```

## What Still Is Not Covered

These areas are still not implemented as release gates:

- retrieval `precision@k` and `recall@k`
- verdict agreement against human-labeled outcome bands
- repeated-run consistency under stochastic sampling
- freshness-window enforcement inside the verifier

Those remain useful next steps, but they are separate from the current grounding and corpus-quality gates.

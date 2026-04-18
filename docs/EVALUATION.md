# Evaluation Strategy

The project should be evaluated as a retrieval system, an analysis system, and a final decision-support system.

## 1. Retrieval Evaluation

Measure whether the system finds the right evidence.

- `precision@k`: How many retrieved chunks are relevant?
- `recall@k`: Did the retriever surface the key evidence at all?
- `MRR` or `nDCG`: Are the best chunks ranked near the top?
- `freshness rate`: Percentage of retrieved evidence within the allowed recency window.

## 2. Analyst Evaluation

Measure whether the specialist agents transform evidence into faithful findings.

- `claim citation coverage`: Share of claims linked to at least one source.
- `unsupported claim rate`: Share of claims lacking evidence.
- `contradiction detection rate`: Whether conflicting evidence is surfaced.
- `schema pass rate`: Whether outputs conform to typed analyst schemas.

## 3. Final Report Evaluation

Measure the usefulness and trustworthiness of the final thesis.

- `verdict agreement`: Agreement with human-labeled verdict bands.
- `reason completeness`: Whether bull, base, and bear cases are all represented.
- `faithfulness`: Human or model-graded score for evidence alignment.
- `consistency`: Stability of final verdict over repeated runs on fixed inputs.

## 4. Human Review Loop

For an initial research workflow, human review matters more than raw automation rate.

Reviewers should score:

- relevance of retrieved evidence
- clarity of the final thesis
- whether major risks were surfaced
- whether the citations would let an analyst verify the claims quickly

## 5. Minimum Viable Golden Set

Create a labeled set of at least:

- 20 ticker-question pairs
- 3 sectors
- a mix of growth, mature, cyclical, and high-volatility names
- known positive and negative events

Each example should include:

- expected key documents
- required issues the system should mention
- unacceptable hallucinations
- a rough verdict band

## 6. Operational Metrics

Track these in every run:

- end-to-end latency
- per-node latency
- token usage by node
- retrieval hit counts
- citation coverage
- verifier failure count

## 7. Release Gates

Before wider use, require:

- graph build passes
- schema conformance above 99%
- citation coverage above target threshold
- unsupported claim rate below threshold
- no critical regression on the golden set

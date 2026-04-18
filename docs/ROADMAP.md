# Roadmap

## Phase 1: Foundation

- Stabilize the workflow contract around typed state and evidence objects.
- Ingest a starter corpus of SEC filings, earnings transcripts, and trusted news.
- Replace generic web search with local retrieval plus source metadata.
- Add report templates that require citations for material claims.

Status:
- completed

## Phase 2: Retrieval Quality

- Introduce chunking by section and earnings-call speaker turns.
- Add hybrid retrieval: metadata filters + BM25 + vectors.
- Add reranking for top-k evidence selection.
- Version corpora and track document freshness.

Status:
- section chunking completed
- speaker-turn chunking completed
- local profile-based retrieval completed
- hybrid retrieval completed
- reranking completed
- neighbor-chunk support completed
- pgvector-native indexing still pending

## Phase 3: Analyst Specialization

- Give each analyst its own retrieval profile and evidence budget.
- Add contradiction tracking between analysts.
- Introduce structured risk taxonomy:
  - valuation risk
  - governance risk
  - balance-sheet risk
  - execution risk
  - sentiment risk

Status:
- analyst-specific retrieval profiles completed
- structured analyst outputs completed
- thesis preparation completed
- contradiction tracking and explicit taxonomy still pending

## Phase 4: Verification and Reliability

- Add a verifier that checks:
  - all verdict-driving claims have citations
  - sources are within freshness limits
  - unsupported claims are flagged
- Add fallback behaviors when retrieval is weak or stale.
- Add observability for latency, token usage, retrieval hit rates, and citation coverage.

Status:
- citation coverage checks completed
- structured finding grounding checks completed
- fail-closed gating completed
- latency and token observability completed
- freshness rules still pending

## Phase 5: Evaluation and Delivery

- Build a golden set across sectors and market regimes.
- Measure precision and recall on retrieval and issue extraction.
- Track thesis consistency under repeated runs.
- Add CI checks for graph build, schema conformance, and regression evals.

Status:
- schema and regression tests partially in place
- golden set and offline evaluation still pending

## Suggested Milestones

1. Hybrid retriever with reranker.
2. Golden set and offline eval dashboard.
3. Add contradiction tracking across analyst outputs.
4. Add freshness-aware verification rules.
5. Add cost estimation and retrieval hit-rate analytics.

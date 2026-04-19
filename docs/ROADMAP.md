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
- Filter off-ticker news and prioritize direct company mentions.

Status:
- section chunking completed
- speaker-turn chunking completed
- local profile-based retrieval completed
- hybrid retrieval completed
- reranking completed
- neighbor-chunk support completed
- off-ticker news filtering completed
- source-quality tiers completed
- pgvector-native indexing still pending
- freshness-aware ranking still pending

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
- deterministic contradiction detection completed
- contradiction review normalization completed
- explicit risk taxonomy still pending

## Phase 4: Verification and Reliability

- Add a verifier that checks:
  - all verdict-driving claims have citations
  - sources are within freshness limits
  - unsupported claims are flagged
- Add fallback behaviors when retrieval is weak or stale.
- Add observability for latency, token usage, retrieval hit rates, and citation coverage.

Status:
- exact `[source:<id>]` citation contract completed
- structured finding grounding checks completed
- fail-closed gating completed
- uncited numeric-claim rejection completed
- malformed citation rejection completed
- single-pass verifier-driven repair completed
- canned end-to-end grounding regressions completed
- latency and token observability completed
- off-ticker retrieval telemetry completed
- freshness rules still pending

## Phase 5: Evaluation and Delivery

- Build a golden set across sectors and market regimes.
- Measure precision and recall on retrieval and issue extraction.
- Track thesis consistency under repeated runs.
- Add CI checks for graph build, schema conformance, and regression evals.
- Add release-gate aggregation for grounding and corpus-quality metrics.

Status:
- deterministic regression tests completed
- 24-case golden set completed
- release-gate aggregation completed
- retrieval `precision@k` and `recall@k` aggregation completed
- CLI release-gate evaluation completed
- roadmap and evaluation docs refreshed
- retrieval relevance labeling for the golden set still pending
- repeated-run consistency evaluation still pending
- CI integration for offline evaluation still pending

## Suggested Milestones

1. Wire the release-gate evaluator into CI so golden-set coverage is enforced automatically.
2. Add labeled retrieval relevance judgments for `precision@k` and `recall@k`.
3. Add freshness-aware verification rules and release thresholds.
4. Replace hardcoded news aliases with an issuer registry and entity-linking pipeline.
5. Add explicit structured risk taxonomy fields through thesis preparation and artifact persistence.

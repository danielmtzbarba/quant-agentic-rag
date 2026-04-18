# AI System Design

## Objective

Build a production-oriented research system for equity analysis that separates:

- corpus ingestion
- retrieval
- specialist analysis
- synthesis
- verification
- operational observability

## Target Architecture

```text
External Sources
  -> Ingestion Pipelines
     - SEC filings
     - earnings transcripts
     - news
     - fundamentals / market data
  -> Raw Storage
  -> Normalization Layer
  -> Chunking + Metadata Enrichment
  -> Vector / Hybrid Retrieval Layer
  -> Specialist Agents
     - fundamentals
     - sentiment
     - risk
  -> Thesis Writer
  -> Verifier
  -> API / UI / downstream consumers
```

## Current Implemented Flow

1. `planner`
2. `fundamentals_retrieval`
3. `fundamentals_corpus_retrieval`
4. `sentiment_corpus_retrieval`
5. `risk_corpus_retrieval`
6. `aggregate_evidence`
7. `fundamentals_analyst`
8. `sentiment_analyst`
9. `risk_analyst`
10. `thesis_preparation`
11. `thesis`
12. `verifier`

Key implementation details:

- specialists no longer share one generic evidence bundle
- analyst outputs are structured, not only free-text summaries
- thesis preparation deterministically maps findings into report sections before synthesis
- verifier compares report citations against analyst `evidence_ids`
- verifier can fail closed when unsupported or partially grounded findings exceed thresholds
- per-node LLM telemetry is captured for planner, analysts, thesis, and verifier
- runtime telemetry includes retries, timeouts, and estimated token cost
- retrieval telemetry includes hit rates and source freshness summaries
- the merged evidence pool is retained for final thesis composition and report-level checks

## Storage Boundaries

- Raw artifacts:
  Provider-native payloads used for reproducibility and reprocessing.
- Normalized documents:
  Source-agnostic records with versioned metadata.
- Retrieval chunks:
  Searchable text units with provenance.
- Structured stores:
  Fundamentals, market data, and operational metrics.

## Implemented Decisions

- Use `sec-edgar-downloader` for EDGAR acquisition.
- Use Alpha Vantage for transcript and news acquisition during prototyping.
- Store SEC raw files on disk under `data/raw/sec`.
- Store transcript raw payloads on disk under `data/raw/transcripts/alpha_vantage`.
- Store news raw payloads on disk under `data/raw/news/alpha_vantage`.
- Normalize filings into `DocumentRecord`.
- Normalize transcripts into `DocumentRecord` with `transcript_turns`.
- Normalize news articles into `DocumentRecord` with publisher and sentiment metadata.
- Chunk section-level content into `EvidenceChunk`.
- Chunk transcripts by speaker turn and preserve speaker metadata on each chunk.
- Chunk news articles into compact article-level chunks with sentiment metadata.
- Keep the current local retriever working by writing SEC chunks as JSONL under `data/chunks`.
- Keep transcript chunks in JSONL under `data/chunks/transcripts`.
- Keep news chunks in JSONL under `data/chunks/news`.
- Use Postgres as a registry layer for ingestion runs, document metadata, and chunk metadata.
- Use analyst-specific retrieval policies:
  - fundamentals -> filings
  - sentiment -> transcripts + news
  - risk -> filings + news + transcripts
- Use structured analyst outputs with:
  - findings
  - evidence ids
  - confidence
  - missing data
- Use a thesis-preparation layer to map findings into:
  - executive summary
  - bull case
  - bear case
  - key risks
  - evidence gaps
- Use deterministic grounding checks in the verifier before LLM-based judgment
- Persist research audit artifacts, verifier metrics, and LLM telemetry in Postgres

## Next Architecture Steps

- Replace local lexical retrieval with hybrid retrieval:
  metadata filters + BM25 + vectors + reranking.
- Move candles to InfluxDB once MetaTrader 5 ingestion is ready.
- Keep SQL or equivalent metadata storage for document registry and versioning.
- Add richer retrieval observability, freshness metrics, and cost estimation.
- Persist vector indexing metadata once the hybrid retriever is introduced.

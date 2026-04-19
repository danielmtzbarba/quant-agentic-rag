# News Ingestion

## Goal

Ingest recent company news from Alpha Vantage into the same corpus pipeline used for filings and transcripts:

- persist raw provider payloads
- normalize each article into a `DocumentRecord`
- create retrieval-ready `EvidenceChunk` records
- persist registry metadata in Postgres under the `rag` schema

## Source

- Provider: Alpha Vantage
- Endpoint shape: `/query?function=NEWS_SENTIMENT&tickers={ticker}&limit={limit}&apikey=...`
- Auth: `VANTAGE_API_KEY`

## Storage Layout

- raw payloads: `data/raw/news/alpha_vantage/{ticker}/latest-limit-{limit}.json`
- normalized documents: `data/normalized/news/{ticker}/`
- retrieval chunks: `data/chunks/news/{ticker}/`

## Normalization

Each article becomes a `DocumentRecord` with:

- `source_type="news"`
- publisher metadata
- timestamp and URL
- sentiment label and score from the provider feed
- ticker relevance metadata:
  - `ticker_relevance_score`
  - `entity_title_match`
  - `entity_body_match`
  - `news_relevance_score`
  - `news_relevance_tier`
  - `source_quality_tier`
- article text assembled from headline, publisher, summary, and sentiment metadata

Articles are now rejected when the requested ticker is present only in provider sentiment metadata
but the title and summary do not directly mention the company or ticker.

## Chunking Strategy

- one chunk per article for now
- preserve `publisher`, `sentiment_label`, and `sentiment_score`
- preserve entity-match and source-quality metadata for retrieval
- keep chunks compact and freshness-oriented for the sentiment and risk agents

## Registry Persistence

When `DATABASE_URL` is configured:

- ingestion runs are tracked in `rag.ingestion_runs`
- normalized news documents are registered in `rag.documents`
- article chunks are registered in `rag.chunks`

## CLI

```bash
uv run stock-agent-rag ingest-news --ticker NVDA --limit 20
```

## Current Limitations

- the current implementation stores provider summaries, not full article bodies
- alias coverage is seeded in code and should be expanded or externalized over time
- direct-mention validation is conservative and may drop borderline sector-context articles

See [NEWS_RELEVANCE_SCORING.md](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/docs/NEWS_RELEVANCE_SCORING.md)
for the current scoring and filtering rules.

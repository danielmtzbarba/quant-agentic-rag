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
- article text assembled from headline, publisher, summary, and sentiment metadata

## Chunking Strategy

- one chunk per article for now
- preserve `publisher`, `sentiment_label`, and `sentiment_score`
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
- source-specific ranking and publisher trust scoring can be added later
- hybrid retrieval and recency-aware reranking are still future work

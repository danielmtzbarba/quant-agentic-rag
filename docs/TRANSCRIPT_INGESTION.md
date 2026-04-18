# Transcript Ingestion

## Goal

Ingest earnings-call transcripts from Alpha Vantage into the same corpus pipeline used for SEC filings:

- persist raw provider payloads
- normalize into `DocumentRecord`
- chunk by speaker turn into `EvidenceChunk`
- register ingestion metadata in Postgres under the `rag` schema

## Source

- Provider: Alpha Vantage
- Endpoint shape: `/query?function=EARNINGS_CALL_TRANSCRIPT&symbol={ticker}&quarter={year}Q{quarter}&apikey=...`
- Auth: `VANTAGE_API_KEY`

## Storage Layout

- raw payloads: `data/raw/transcripts/alpha_vantage/{ticker}/{year}/q{quarter}.json`
- normalized transcript documents: `data/normalized/transcripts/{ticker}/`
- retrieval chunks: `data/chunks/transcripts/{ticker}/`

## Normalization

Each transcript becomes a `DocumentRecord` with:

- `source_type="transcript"`
- provider metadata
- timestamp and title
- normalized transcript text
- `transcript_turns` extracted from speaker headers

## Chunking Strategy

- parse the transcript into speaker turns
- preserve `speaker` and `speaker_role`
- prepend speaker context to each chunk before splitting
- split long turns into overlapping sub-chunks for retrieval safety

This keeps retrieval grounded in both the statement content and the identity of the speaker.

## Registry Persistence

When `DATABASE_URL` is configured:

- ingestion runs are tracked in `rag.ingestion_runs`
- transcript documents are registered in `rag.documents`
- speaker-aware chunks are registered in `rag.chunks`

## CLI

```bash
uv run stock-agent-rag ingest-transcript --ticker NVDA --year 2025 --quarter 4
```

## Current Limitations

- the current implementation fetches one transcript per explicit `ticker/year/quarter`
- speaker parsing is heuristic and should be hardened against more transcript formats
- transcript listing and backfill workflows can be added after the first production slice is stable

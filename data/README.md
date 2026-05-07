# Data Layout

The ingestion pipeline writes corpus artifacts under this top-level `data/` directory.

Current layout:

- `data/raw/`
  Provider-native payloads and downloaded filings.
- `data/normalized/`
  Document-level normalized JSON records with versioned metadata.
- `data/chunks/`
  Retrieval-ready JSONL chunk files used by the current local retriever.

SEC filings are written under:

- `data/raw/sec/`
- `data/normalized/sec/<TICKER>/<FORM_TYPE>/`
- `data/chunks/sec/<TICKER>/<FORM_TYPE>/`

When `DATABASE_URL` is configured, metadata about these artifacts is also written
to Postgres tables for registry and observability.

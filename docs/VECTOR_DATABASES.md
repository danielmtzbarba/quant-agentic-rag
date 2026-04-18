# Vector Databases And Retrieval

## Role In The System

The vector store is not the source of truth. It is the retrieval acceleration layer over versioned chunks.

In the current implementation, ingestion and indexing are separate:

- ingestion persists documents and chunks
- indexing persists embeddings in Postgres-backed `chunk_embeddings`
- retrieval joins chunk metadata with the embedding index
- Postgres / Supabase uses a native `pgvector` column and cosine-distance search path
- SQLite and fallback paths use the persisted JSON embedding payload

## Recommended Contract

Each indexed chunk should preserve:

- `chunk_id`
- `document_id`
- `ticker`
- `document_type`
- `provider`
- `form_type`
- `section`
- `published_at`
- `metadata_version`

## Retrieval Strategy

1. filter by ticker and document type
2. apply freshness constraints
3. retrieve with lexical + vector methods
4. rerank top candidates
5. pass evidence-limited context to specialist agents

## Storage Recommendation

- prototype:
  local JSONL chunks plus simple lexical retrieval
- current:
  Postgres-backed chunk registry plus dedicated embedding index tables and hybrid retrieval
- next:
  move semantic scoring fully into pgvector indexes for larger corpora
- production:
  hybrid retrieval with reranking, freshness enforcement, and evaluation hooks

## Current Retrieval Stack

1. prefilter chunks by metadata:
   - ticker
   - document type
   - form type
   - section
   - recency
   - publisher / speaker role
2. keep document-aware chunk units:
   - filing sections
   - transcript speaker turns
   - news articles
3. rewrite the user query for retrieval and decompose it into finance-specific subqueries
4. run lexical and semantic retrieval across the subquery set
5. fuse with reciprocal rank fusion (`RRF`)
6. rerank the fused top candidates
7. apply freshness-aware scoring and source-diversity controls
8. return the primary chunks plus limited neighbor support

## Structured Data Boundary

Do not put fundamentals ratios or candle data into the vector database as the primary storage.

- candles:
  InfluxDB later, `yfinance` for prototyping now
- fundamentals:
  SQL or structured metadata store

create table if not exists rag.indexing_runs (
    run_id varchar(128) primary key,
    ticker varchar(16),
    embedding_model varchar(128) not null,
    status varchar(32) not null,
    indexed_chunks integer not null default 0,
    skipped_chunks integer not null default 0,
    started_at timestamptz not null,
    completed_at timestamptz,
    error_message text
);

create index if not exists ix_rag_indexing_runs_ticker
    on rag.indexing_runs (ticker);
create index if not exists ix_rag_indexing_runs_embedding_model
    on rag.indexing_runs (embedding_model);
create index if not exists ix_rag_indexing_runs_status
    on rag.indexing_runs (status);

create table if not exists rag.chunk_embeddings (
    chunk_id varchar(255) primary key,
    document_id varchar(255) not null,
    ticker varchar(16) not null,
    embedding_model varchar(128) not null,
    embedding_dimensions integer not null,
    embedding_json jsonb not null,
    indexed_at timestamptz not null
);

create index if not exists ix_rag_chunk_embeddings_document_id
    on rag.chunk_embeddings (document_id);
create index if not exists ix_rag_chunk_embeddings_ticker
    on rag.chunk_embeddings (ticker);
create index if not exists ix_rag_chunk_embeddings_embedding_model
    on rag.chunk_embeddings (embedding_model);
create index if not exists ix_rag_chunk_embeddings_indexed_at
    on rag.chunk_embeddings (indexed_at);

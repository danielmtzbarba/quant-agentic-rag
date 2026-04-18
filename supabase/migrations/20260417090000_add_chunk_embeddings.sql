alter table if exists rag.chunks
    add column if not exists embedding_json jsonb,
    add column if not exists embedding_model varchar(128),
    add column if not exists embedded_at timestamptz;

create index if not exists ix_rag_chunks_embedding_model
    on rag.chunks (embedding_model);

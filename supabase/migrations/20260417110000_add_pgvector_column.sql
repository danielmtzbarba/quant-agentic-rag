create schema if not exists extensions;
create extension if not exists vector with schema extensions;

alter table if exists rag.chunk_embeddings
    add column if not exists embedding_vector extensions.vector(1536);

create index if not exists ix_rag_chunk_embeddings_embedding_vector_hnsw
    on rag.chunk_embeddings
    using hnsw (embedding_vector extensions.vector_cosine_ops);

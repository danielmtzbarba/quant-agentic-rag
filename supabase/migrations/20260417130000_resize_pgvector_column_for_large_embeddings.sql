create schema if not exists extensions;
create extension if not exists vector with schema extensions;

drop index if exists rag.ix_rag_chunk_embeddings_embedding_vector_hnsw;

alter table if exists rag.chunk_embeddings
    drop column if exists embedding_vector;

alter table if exists rag.chunk_embeddings
    add column embedding_vector extensions.vector(3072);

-- pgvector HNSW indexes do not support 3072-dimension `vector` columns.
-- Keep the native vector column for exact cosine search and revisit indexing
-- later with a lower-dimensional embedding, halfvec, or quantization path.

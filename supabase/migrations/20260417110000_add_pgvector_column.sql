create schema if not exists extensions;
create extension if not exists vector with schema extensions;

alter table if exists rag.chunk_embeddings
    add column if not exists embedding_vector extensions.vector(1536);

do $$
declare
    column_type text;
    embedding_dims integer;
begin
    select pg_catalog.format_type(a.atttypid, a.atttypmod)
      into column_type
      from pg_catalog.pg_attribute a
      join pg_catalog.pg_class c
        on c.oid = a.attrelid
      join pg_catalog.pg_namespace n
        on n.oid = c.relnamespace
     where n.nspname = 'rag'
       and c.relname = 'chunk_embeddings'
       and a.attname = 'embedding_vector'
       and a.attisdropped = false;

    if column_type is null then
        raise notice 'rag.chunk_embeddings.embedding_vector not found; skipping hnsw index creation.';
        return;
    end if;

    if column_type ~ 'vector\(([0-9]+)\)' then
        embedding_dims := substring(column_type from 'vector\(([0-9]+)\)')::integer;
    end if;

    if embedding_dims is null then
        raise notice 'Could not determine embedding_vector dimensions from %, skipping hnsw index creation.', column_type;
        return;
    end if;

    if embedding_dims > 2000 then
        raise notice 'Skipping hnsw index creation for rag.chunk_embeddings.embedding_vector because dimensions=% exceed pgvector hnsw limit.', embedding_dims;
        return;
    end if;

    execute $sql$
        create index if not exists ix_rag_chunk_embeddings_embedding_vector_hnsw
            on rag.chunk_embeddings
            using hnsw (embedding_vector extensions.vector_cosine_ops)
    $sql$;
end
$$;

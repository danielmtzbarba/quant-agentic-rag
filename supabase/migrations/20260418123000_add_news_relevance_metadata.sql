alter table if exists rag.documents
    add column if not exists ticker_relevance_score double precision,
    add column if not exists entity_title_match boolean,
    add column if not exists entity_body_match boolean,
    add column if not exists news_relevance_score double precision,
    add column if not exists news_relevance_tier varchar(32),
    add column if not exists source_quality_tier varchar(32);

alter table if exists rag.chunks
    add column if not exists ticker_relevance_score double precision,
    add column if not exists entity_title_match boolean,
    add column if not exists entity_body_match boolean,
    add column if not exists news_relevance_score double precision,
    add column if not exists news_relevance_tier varchar(32),
    add column if not exists source_quality_tier varchar(32);

create index if not exists ix_rag_documents_news_relevance_score
    on rag.documents (news_relevance_score desc);

create index if not exists ix_rag_documents_news_relevance_tier
    on rag.documents (news_relevance_tier);

create index if not exists ix_rag_documents_source_quality_tier
    on rag.documents (source_quality_tier);

create index if not exists ix_rag_chunks_news_relevance_score
    on rag.chunks (news_relevance_score desc);

create index if not exists ix_rag_chunks_news_relevance_tier
    on rag.chunks (news_relevance_tier);

create index if not exists ix_rag_chunks_source_quality_tier
    on rag.chunks (source_quality_tier);

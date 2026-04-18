create table if not exists rag.thesis_artifacts (
    thesis_id varchar(128) primary key,
    run_id varchar(128) not null,
    ticker varchar(16) not null,
    question text not null,
    artifact_version varchar(32) not null default '1.0',
    created_at timestamptz not null default now(),
    storage_provider varchar(64) not null,
    bucket varchar(255) not null,
    object_key varchar(1000) not null,
    content_type varchar(128) not null default 'text/markdown',
    markdown_path varchar(1000),
    markdown_checksum varchar(64) not null,
    object_etag varchar(255),
    status varchar(32) not null default 'completed',
    verification_status varchar(32),
    deterministic_verifier_status varchar(32),
    model_name varchar(128),
    embedding_model varchar(128),
    retrieved_source_count integer not null default 0,
    cited_source_count integer not null default 0,
    citation_coverage double precision,
    structured_findings_count integer not null default 0,
    unsupported_findings_count integer not null default 0,
    partially_grounded_findings_count integer not null default 0,
    contradictions_count integer not null default 0,
    latency_ms double precision,
    estimated_cost_usd double precision,
    thesis_word_count integer not null default 0,
    thesis_hash varchar(64) not null,
    top_source_ids_json jsonb not null default '[]'::jsonb,
    contradictions_json jsonb,
    retrieval_metrics_json jsonb,
    verification_metrics_json jsonb,
    runtime_metrics_json jsonb,
    tags_json jsonb not null default '[]'::jsonb
);

create index if not exists ix_rag_thesis_artifacts_run_id
    on rag.thesis_artifacts (run_id);

create index if not exists ix_rag_thesis_artifacts_ticker_created_at
    on rag.thesis_artifacts (ticker, created_at desc);

create index if not exists ix_rag_thesis_artifacts_verification_status
    on rag.thesis_artifacts (verification_status);

create index if not exists ix_rag_thesis_artifacts_thesis_hash
    on rag.thesis_artifacts (thesis_hash);

create unique index if not exists ux_rag_thesis_artifacts_bucket_object_key
    on rag.thesis_artifacts (bucket, object_key);

create schema if not exists rag;

create table if not exists rag.ingestion_runs (
    run_id varchar(128) primary key,
    source_type varchar(32) not null,
    ticker varchar(16) not null,
    form_type varchar(16),
    status varchar(32) not null,
    metadata_version varchar(32) not null,
    processed_documents integer not null default 0,
    chunk_count integer not null default 0,
    started_at timestamptz not null,
    completed_at timestamptz,
    error_message text
);

create index if not exists ix_rag_ingestion_runs_source_type
    on rag.ingestion_runs (source_type);
create index if not exists ix_rag_ingestion_runs_ticker
    on rag.ingestion_runs (ticker);
create index if not exists ix_rag_ingestion_runs_form_type
    on rag.ingestion_runs (form_type);
create index if not exists ix_rag_ingestion_runs_status
    on rag.ingestion_runs (status);

create table if not exists rag.documents (
    document_id varchar(255) primary key,
    source_type varchar(32) not null,
    ticker varchar(16) not null,
    title varchar(500) not null,
    provider varchar(128) not null,
    form_type varchar(16),
    published_at timestamptz,
    as_of_date date,
    source_url varchar(1000),
    accession_number varchar(32),
    cik varchar(32),
    metadata_version varchar(32) not null,
    raw_checksum varchar(64) not null,
    raw_path varchar(1000) not null,
    normalized_path varchar(1000),
    cleaned_text text not null,
    publisher varchar(255),
    sentiment_label varchar(64),
    sentiment_score double precision,
    section_count integer not null default 0,
    sections_json jsonb not null default '[]'::jsonb,
    transcript_turn_count integer not null default 0,
    transcript_turns_json jsonb not null default '[]'::jsonb
);

create index if not exists ix_rag_documents_source_type
    on rag.documents (source_type);
create index if not exists ix_rag_documents_ticker
    on rag.documents (ticker);
create index if not exists ix_rag_documents_provider
    on rag.documents (provider);
create index if not exists ix_rag_documents_form_type
    on rag.documents (form_type);
create index if not exists ix_rag_documents_metadata_version
    on rag.documents (metadata_version);
create index if not exists ix_rag_documents_accession_number
    on rag.documents (accession_number);
create index if not exists ix_rag_documents_cik
    on rag.documents (cik);
create index if not exists ix_rag_documents_publisher
    on rag.documents (publisher);
create index if not exists ix_rag_documents_sentiment_label
    on rag.documents (sentiment_label);

create table if not exists rag.chunks (
    chunk_id varchar(255) primary key,
    source_id varchar(255) not null,
    document_id varchar(255) not null,
    ticker varchar(16) not null,
    title varchar(500) not null,
    content text not null,
    document_type varchar(32) not null,
    provider varchar(128) not null,
    form_type varchar(16),
    section varchar(128),
    source_url varchar(1000),
    published_at timestamptz,
    accession_number varchar(32),
    chunk_index integer not null,
    metadata_version varchar(32) not null,
    score double precision not null default 0.0,
    chunk_path varchar(1000),
    speaker varchar(255),
    speaker_role varchar(255),
    publisher varchar(255),
    sentiment_label varchar(64),
    sentiment_score double precision
);

create index if not exists ix_rag_chunks_source_id
    on rag.chunks (source_id);
create index if not exists ix_rag_chunks_document_id
    on rag.chunks (document_id);
create index if not exists ix_rag_chunks_ticker
    on rag.chunks (ticker);
create index if not exists ix_rag_chunks_document_type
    on rag.chunks (document_type);
create index if not exists ix_rag_chunks_provider
    on rag.chunks (provider);
create index if not exists ix_rag_chunks_form_type
    on rag.chunks (form_type);
create index if not exists ix_rag_chunks_section
    on rag.chunks (section);
create index if not exists ix_rag_chunks_accession_number
    on rag.chunks (accession_number);
create index if not exists ix_rag_chunks_metadata_version
    on rag.chunks (metadata_version);
create index if not exists ix_rag_chunks_speaker
    on rag.chunks (speaker);
create index if not exists ix_rag_chunks_speaker_role
    on rag.chunks (speaker_role);
create index if not exists ix_rag_chunks_publisher
    on rag.chunks (publisher);
create index if not exists ix_rag_chunks_sentiment_label
    on rag.chunks (sentiment_label);

create table if not exists rag.source_registry (
    source_key varchar(255) primary key,
    source_type varchar(32) not null,
    ticker varchar(16) not null,
    provider varchar(128) not null,
    latest_document_id varchar(255),
    latest_published_at timestamptz,
    metadata_version varchar(32) not null,
    active boolean not null default true
);

create index if not exists ix_rag_source_registry_source_type
    on rag.source_registry (source_type);
create index if not exists ix_rag_source_registry_ticker
    on rag.source_registry (ticker);
create index if not exists ix_rag_source_registry_provider
    on rag.source_registry (provider);
create index if not exists ix_rag_source_registry_latest_document_id
    on rag.source_registry (latest_document_id);

create table if not exists rag.research_runs (
    run_id varchar(128) primary key,
    ticker varchar(16) not null,
    question text not null,
    status varchar(32) not null,
    verification_status varchar(32),
    started_at timestamptz not null,
    completed_at timestamptz,
    latency_ms double precision,
    plan text,
    report text,
    verification_summary text,
    retrieved_source_ids_json jsonb not null default '[]'::jsonb,
    node_metrics_json jsonb,
    token_usage_json jsonb,
    model_metadata_json jsonb,
    runtime_metrics_json jsonb,
    retrieval_metrics_json jsonb,
    estimated_cost_usd double precision,
    fundamentals_analysis_json jsonb,
    sentiment_analysis_json jsonb,
    risk_analysis_json jsonb,
    thesis_preparation_json jsonb,
    verification_metrics_json jsonb,
    error_message text
);

create index if not exists ix_rag_research_runs_ticker
    on rag.research_runs (ticker);
create index if not exists ix_rag_research_runs_status
    on rag.research_runs (status);
create index if not exists ix_rag_research_runs_verification_status
    on rag.research_runs (verification_status);

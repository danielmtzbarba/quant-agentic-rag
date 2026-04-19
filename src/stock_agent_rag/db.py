from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import Text as SqlText
from sqlalchemy.types import UserDefinedType

from .config import Settings, get_settings


class Base(DeclarativeBase):
    metadata = MetaData()


class VectorType(UserDefinedType):
    cache_ok = True

    def __init__(self, dimensions: int, *, schema: str = "extensions") -> None:
        self.dimensions = dimensions
        self.schema = schema

    def get_col_spec(self, **_kw) -> str:
        return f"{self.schema}.vector({self.dimensions})"


class IngestionRunORM(Base):
    __tablename__ = "ingestion_runs"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    form_type: Mapped[str | None] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    metadata_version: Mapped[str] = mapped_column(String(32))
    processed_documents: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)


class IndexingRunORM(Base):
    __tablename__ = "indexing_runs"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    ticker: Mapped[str | None] = mapped_column(String(16), index=True)
    embedding_model: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    indexed_chunks: Mapped[int] = mapped_column(Integer, default=0)
    skipped_chunks: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)


class DocumentORM(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    title: Mapped[str] = mapped_column(String(500))
    provider: Mapped[str] = mapped_column(String(128), index=True)
    form_type: Mapped[str | None] = mapped_column(String(16), index=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    as_of_date: Mapped[date | None] = mapped_column(Date)
    source_url: Mapped[str | None] = mapped_column(String(1000))
    accession_number: Mapped[str | None] = mapped_column(String(32), index=True)
    cik: Mapped[str | None] = mapped_column(String(32), index=True)
    metadata_version: Mapped[str] = mapped_column(String(32), index=True)
    raw_checksum: Mapped[str] = mapped_column(String(64))
    raw_path: Mapped[str] = mapped_column(String(1000))
    normalized_path: Mapped[str | None] = mapped_column(String(1000))
    cleaned_text: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(String(255), index=True)
    sentiment_label: Mapped[str | None] = mapped_column(String(64), index=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float)
    ticker_relevance_score: Mapped[float | None] = mapped_column(Float)
    entity_title_match: Mapped[bool | None] = mapped_column(Boolean)
    entity_body_match: Mapped[bool | None] = mapped_column(Boolean)
    news_relevance_score: Mapped[float | None] = mapped_column(Float, index=True)
    news_relevance_tier: Mapped[str | None] = mapped_column(String(32), index=True)
    source_quality_tier: Mapped[str | None] = mapped_column(String(32), index=True)
    section_count: Mapped[int] = mapped_column(Integer, default=0)
    sections_json: Mapped[list[dict]] = mapped_column(JSON, default=list)
    transcript_turn_count: Mapped[int] = mapped_column(Integer, default=0)
    transcript_turns_json: Mapped[list[dict]] = mapped_column(JSON, default=list)


class ChunkORM(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(255), index=True)
    document_id: Mapped[str] = mapped_column(String(255), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)
    document_type: Mapped[str] = mapped_column(String(32), index=True)
    provider: Mapped[str] = mapped_column(String(128), index=True)
    form_type: Mapped[str | None] = mapped_column(String(16), index=True)
    section: Mapped[str | None] = mapped_column(String(128), index=True)
    source_url: Mapped[str | None] = mapped_column(String(1000))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    accession_number: Mapped[str | None] = mapped_column(String(32), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    metadata_version: Mapped[str] = mapped_column(String(32), index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    chunk_path: Mapped[str | None] = mapped_column(String(1000))
    speaker: Mapped[str | None] = mapped_column(String(255), index=True)
    speaker_role: Mapped[str | None] = mapped_column(String(255), index=True)
    publisher: Mapped[str | None] = mapped_column(String(255), index=True)
    sentiment_label: Mapped[str | None] = mapped_column(String(64), index=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float)
    ticker_relevance_score: Mapped[float | None] = mapped_column(Float)
    entity_title_match: Mapped[bool | None] = mapped_column(Boolean)
    entity_body_match: Mapped[bool | None] = mapped_column(Boolean)
    news_relevance_score: Mapped[float | None] = mapped_column(Float, index=True)
    news_relevance_tier: Mapped[str | None] = mapped_column(String(32), index=True)
    source_quality_tier: Mapped[str | None] = mapped_column(String(32), index=True)


class ChunkEmbeddingORM(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(255), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    embedding_model: Mapped[str] = mapped_column(String(128), index=True)
    embedding_dimensions: Mapped[int] = mapped_column(Integer)
    embedding_json: Mapped[list[float]] = mapped_column(JSON)
    embedding_vector: Mapped[str | None] = mapped_column(
        SqlText().with_variant(VectorType(3072), "postgresql"),
        nullable=True,
    )
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class SourceRegistryORM(Base):
    __tablename__ = "source_registry"

    source_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    provider: Mapped[str] = mapped_column(String(128), index=True)
    latest_document_id: Mapped[str | None] = mapped_column(String(255), index=True)
    latest_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_version: Mapped[str] = mapped_column(String(32))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class ResearchRunORM(Base):
    __tablename__ = "research_runs"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), index=True)
    verification_status: Mapped[str | None] = mapped_column(String(32), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    plan: Mapped[str | None] = mapped_column(Text)
    report: Mapped[str | None] = mapped_column(Text)
    verification_summary: Mapped[str | None] = mapped_column(Text)
    retrieved_source_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    node_metrics_json: Mapped[dict | None] = mapped_column(JSON)
    token_usage_json: Mapped[dict | None] = mapped_column(JSON)
    model_metadata_json: Mapped[dict | None] = mapped_column(JSON)
    runtime_metrics_json: Mapped[dict | None] = mapped_column(JSON)
    retrieval_metrics_json: Mapped[dict | None] = mapped_column(JSON)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)
    fundamentals_analysis_json: Mapped[dict | None] = mapped_column(JSON)
    sentiment_analysis_json: Mapped[dict | None] = mapped_column(JSON)
    risk_analysis_json: Mapped[dict | None] = mapped_column(JSON)
    thesis_preparation_json: Mapped[dict | None] = mapped_column(JSON)
    verification_metrics_json: Mapped[dict | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)


class ThesisArtifactORM(Base):
    __tablename__ = "thesis_artifacts"

    thesis_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    question: Mapped[str] = mapped_column(Text)
    artifact_version: Mapped[str] = mapped_column(String(32), default="1.0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    storage_provider: Mapped[str] = mapped_column(String(64))
    bucket: Mapped[str] = mapped_column(String(255))
    object_key: Mapped[str] = mapped_column(String(1000))
    content_type: Mapped[str] = mapped_column(String(128), default="text/markdown")
    markdown_path: Mapped[str | None] = mapped_column(String(1000))
    markdown_checksum: Mapped[str] = mapped_column(String(64))
    object_etag: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    verification_status: Mapped[str | None] = mapped_column(String(32), index=True)
    deterministic_verifier_status: Mapped[str | None] = mapped_column(String(32))
    model_name: Mapped[str | None] = mapped_column(String(128))
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    retrieved_source_count: Mapped[int] = mapped_column(Integer, default=0)
    cited_source_count: Mapped[int] = mapped_column(Integer, default=0)
    citation_coverage: Mapped[float | None] = mapped_column(Float)
    structured_findings_count: Mapped[int] = mapped_column(Integer, default=0)
    unsupported_findings_count: Mapped[int] = mapped_column(Integer, default=0)
    partially_grounded_findings_count: Mapped[int] = mapped_column(Integer, default=0)
    contradictions_count: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)
    thesis_word_count: Mapped[int] = mapped_column(Integer, default=0)
    thesis_hash: Mapped[str] = mapped_column(String(64), index=True)
    top_source_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    contradictions_json: Mapped[list[dict] | None] = mapped_column(JSON)
    retrieval_metrics_json: Mapped[dict | None] = mapped_column(JSON)
    verification_metrics_json: Mapped[dict | None] = mapped_column(JSON)
    runtime_metrics_json: Mapped[dict | None] = mapped_column(JSON)
    tags_json: Mapped[list[str]] = mapped_column(JSON, default=list)


def _normalize_database_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _is_sqlite_url(url: str) -> bool:
    return url.startswith("sqlite")


@lru_cache(maxsize=1)
def get_engine():
    settings = get_settings()
    if not settings.db_enabled:
        raise RuntimeError("DATABASE_URL is not configured.")
    normalized_url = _normalize_database_url(settings.database_url.strip())
    connect_args: dict[str, object] = {}
    if not _is_sqlite_url(normalized_url) and normalized_url.startswith("postgresql+psycopg://"):
        # Supabase pooler / PgBouncer does not support psycopg prepared statements reliably.
        connect_args["prepare_threshold"] = None
    engine = create_engine(
        normalized_url,
        echo=settings.db_echo,
        future=True,
        connect_args=connect_args,
    )

    if not _is_sqlite_url(normalized_url):
        schema = settings.db_schema
        engine = engine.execution_options(schema_translate_map={None: schema})

        @event.listens_for(engine, "connect")
        def _set_search_path(dbapi_connection, _connection_record) -> None:
            with dbapi_connection.cursor() as cursor:
                cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                cursor.execute(f'SET search_path TO "{schema}"')

    return engine


@lru_cache(maxsize=1)
def get_session_factory():
    return sessionmaker(bind=get_engine(), class_=Session, expire_on_commit=False)


def get_db_session() -> Session:
    return get_session_factory()()


def initialize_database(settings: Settings | None = None) -> None:
    resolved = settings or get_settings()
    if not resolved.db_enabled:
        raise RuntimeError("DATABASE_URL is not configured.")
    engine = get_engine()
    normalized_url = _normalize_database_url(resolved.database_url.strip())
    if not _is_sqlite_url(normalized_url):
        with engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{resolved.db_schema}"'))
    Base.metadata.create_all(engine)

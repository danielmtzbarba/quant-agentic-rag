from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"

# Explicitly load the project .env so CLI entrypoints keep working even when the
# current working directory is not the repo root.
load_dotenv(dotenv_path=DOTENV_PATH, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="stock-agent-rag", alias="APP_NAME")
    app_env: str = Field(default="local", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="auto", alias="LOG_FORMAT")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    project_id: str | None = Field(default=None, alias="PROJECT_ID")
    model_name: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    thesis_storage_provider: str = Field(default="local", alias="THESIS_STORAGE_PROVIDER")
    thesis_artifact_bucket: str = Field(
        default="thesis-artifacts",
        alias="THESIS_ARTIFACT_BUCKET",
    )
    thesis_artifact_local_mirror: bool = Field(
        default=True,
        alias="THESIS_ARTIFACT_LOCAL_MIRROR",
    )
    thesis_artifact_base_dir: Path = Field(
        default=Path("./data/reports"),
        alias="THESIS_ARTIFACT_BASE_DIR",
    )
    aws_region: str | None = Field(default=None, alias="AWS_REGION")
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_session_token: str | None = Field(default=None, alias="AWS_SESSION_TOKEN")
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    s3_force_path_style: bool = Field(default=False, alias="S3_FORCE_PATH_STYLE")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    corpus_dir: Path = Field(default=Path("./data/chunks"), alias="RAG_CORPUS_DIR")
    default_top_k: int = Field(default=4, alias="DEFAULT_TOP_K")
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    db_echo: bool = Field(default=False, alias="DB_ECHO")
    db_schema: str = Field(default="rag", alias="DB_SCHEMA")
    sec_company_name: str = Field(default="stock-agent-rag", alias="SEC_COMPANY_NAME")
    sec_email_address: str = Field(default="you@example.com", alias="SEC_EMAIL_ADDRESS")
    sec_metadata_version: str = Field(default="1.0", alias="SEC_METADATA_VERSION")
    vantage_api_key: str | None = Field(default=None, alias="VANTAGE_API_KEY")
    vantage_base_url: str = Field(
        default="https://www.alphavantage.co/query",
        alias="VANTAGE_BASE_URL",
    )
    transcript_metadata_version: str = Field(default="1.0", alias="TRANSCRIPT_METADATA_VERSION")
    news_metadata_version: str = Field(default="1.0", alias="NEWS_METADATA_VERSION")
    # Transcript downloader tuning. Keep defaults conservative for reliability in prod.
    transcript_http_timeout_s: float = Field(default=30.0, alias="TRANSCRIPT_HTTP_TIMEOUT_S")
    transcript_http_max_retries: int = Field(default=3, alias="TRANSCRIPT_HTTP_MAX_RETRIES")
    transcript_http_backoff_base_s: float = Field(
        default=0.75, alias="TRANSCRIPT_HTTP_BACKOFF_BASE_S"
    )
    transcript_http_backoff_max_s: float = Field(
        default=10.0, alias="TRANSCRIPT_HTTP_BACKOFF_MAX_S"
    )
    embedding_model_name: str = Field(
        default="text-embedding-3-large",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    embedding_dimensions: int = Field(
        default=3072,
        alias="OPENAI_EMBEDDING_DIMENSIONS",
    )
    reranker_model_name: str = Field(
        default="gpt-4o-mini",
        alias="OPENAI_RERANKER_MODEL",
    )
    retrieval_candidate_pool: int = Field(default=48, alias="RETRIEVAL_CANDIDATE_POOL")
    retrieval_rrf_k: int = Field(default=60, alias="RETRIEVAL_RRF_K")
    retrieval_neighbor_window: int = Field(default=1, alias="RETRIEVAL_NEIGHBOR_WINDOW")
    retrieval_neighbor_limit: int = Field(default=2, alias="RETRIEVAL_NEIGHBOR_LIMIT")
    retrieval_rerank_top_n: int = Field(default=24, alias="RETRIEVAL_RERANK_TOP_N")
    retrieval_embedding_batch_size: int = Field(
        default=32,
        alias="RETRIEVAL_EMBEDDING_BATCH_SIZE",
    )
    retrieval_query_plan_limit: int = Field(default=4, alias="RETRIEVAL_QUERY_PLAN_LIMIT")
    retrieval_max_per_document: int = Field(default=2, alias="RETRIEVAL_MAX_PER_DOCUMENT")
    retrieval_max_news_chunks: int = Field(default=2, alias="RETRIEVAL_MAX_NEWS_CHUNKS")
    retrieval_max_transcript_chunks: int = Field(
        default=3,
        alias="RETRIEVAL_MAX_TRANSCRIPT_CHUNKS",
    )
    retrieval_max_filing_chunks: int = Field(default=3, alias="RETRIEVAL_MAX_FILING_CHUNKS")
    retrieval_sentiment_recency_days: int = Field(
        default=45,
        alias="RETRIEVAL_SENTIMENT_RECENCY_DAYS",
    )
    retrieval_risk_recency_days: int = Field(
        default=120,
        alias="RETRIEVAL_RISK_RECENCY_DAYS",
    )
    verifier_max_unsupported_findings: int = Field(
        default=0,
        alias="VERIFIER_MAX_UNSUPPORTED_FINDINGS",
    )
    verifier_max_partially_grounded_findings: int = Field(
        default=1,
        alias="VERIFIER_MAX_PARTIALLY_GROUNDED_FINDINGS",
    )

    @property
    def is_local(self) -> bool:
        return self.app_env.lower() in {"local", "dev", "development"}

    @property
    def resolved_log_format(self) -> str:
        if self.log_format.lower() == "auto":
            return "rich" if self.is_local else "logfmt"
        return self.log_format.lower()

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def normalized_data_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def chunked_data_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def sec_raw_dir(self) -> Path:
        return self.raw_data_dir / "sec"

    @property
    def transcript_raw_dir(self) -> Path:
        return self.raw_data_dir / "transcripts"

    @property
    def news_raw_dir(self) -> Path:
        return self.raw_data_dir / "news"

    @property
    def db_enabled(self) -> bool:
        return bool(self.database_url and self.database_url.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir = settings.data_dir.resolve()
    settings.corpus_dir = settings.corpus_dir.resolve()
    settings.thesis_artifact_base_dir = settings.thesis_artifact_base_dir.resolve()
    return settings

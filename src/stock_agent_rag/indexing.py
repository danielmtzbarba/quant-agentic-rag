from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .db import ChunkEmbeddingORM, ChunkORM, IndexingRunORM
from .logging import get_logger
from .retrieval import EmbeddingProvider, OpenAIEmbeddingProvider

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexingSummary:
    run_id: str
    ticker: str | None
    embedding_model: str
    indexed_chunks: int
    skipped_chunks: int


class ChunkIndexingService:
    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embedding_provider = embedding_provider or OpenAIEmbeddingProvider(self.settings)

    def index_chunks(
        self,
        *,
        ticker: str | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> IndexingSummary:
        run_id = self._create_run(ticker=ticker)
        indexed_chunks = 0
        skipped_chunks = 0

        try:
            if not force:
                skipped_chunks = self._count_already_indexed_chunks(ticker=ticker)
            rows = list(self._select_chunks(ticker=ticker, limit=limit, force=force))
            batch_size = max(int(self.settings.retrieval_embedding_batch_size), 1)

            logger.info(
                "chunk indexing started",
                extra={
                    "run_id": run_id,
                    "ticker": ticker.upper() if ticker else None,
                    "force": force,
                    "selected_chunks": len(rows),
                    "embedding_model": self.settings.embedding_model_name,
                },
            )

            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                texts = [self._embedding_text_from_chunk(row) for row in batch]
                embeddings = self.embedding_provider.embed_documents(texts)
                now = datetime.now(UTC)
                expected_dimensions = int(
                    getattr(
                        self.settings,
                        "embedding_dimensions",
                        len(embeddings[0]) if embeddings else 0,
                    )
                )
                for row, embedding in zip(batch, embeddings, strict=False):
                    if len(embedding) != expected_dimensions:
                        raise RuntimeError(
                            "Embedding dimensions do not match configured pgvector dimensions: "
                            f"expected={expected_dimensions} actual={len(embedding)}"
                        )
                    existing = self.session.get(ChunkEmbeddingORM, row.chunk_id)
                    payload = ChunkEmbeddingORM(
                        chunk_id=row.chunk_id,
                        document_id=row.document_id,
                        ticker=row.ticker,
                        embedding_model=self.settings.embedding_model_name,
                        embedding_dimensions=len(embedding),
                        embedding_json=embedding,
                        embedding_vector=_vector_literal(embedding),
                        indexed_at=now,
                    )
                    if existing is None:
                        self.session.add(payload)
                    else:
                        existing.document_id = payload.document_id
                        existing.ticker = payload.ticker
                        existing.embedding_model = payload.embedding_model
                        existing.embedding_dimensions = payload.embedding_dimensions
                        existing.embedding_json = payload.embedding_json
                        existing.embedding_vector = payload.embedding_vector
                        existing.indexed_at = payload.indexed_at
                    indexed_chunks += 1
                self.session.commit()
                logger.info(
                    "chunk indexing batch completed",
                    extra={
                        "run_id": run_id,
                        "batch_size": len(batch),
                        "indexed_chunks": indexed_chunks,
                    },
                )

            self._complete_run(
                run_id=run_id,
                indexed_chunks=indexed_chunks,
                skipped_chunks=skipped_chunks,
            )
            logger.info(
                "chunk indexing completed",
                extra={
                    "run_id": run_id,
                    "ticker": ticker.upper() if ticker else None,
                    "indexed_chunks": indexed_chunks,
                    "skipped_chunks": skipped_chunks,
                    "embedding_model": self.settings.embedding_model_name,
                },
            )
        except Exception as exc:
            self._complete_run(
                run_id=run_id,
                indexed_chunks=indexed_chunks,
                skipped_chunks=skipped_chunks,
                error_message=str(exc),
            )
            logger.error(
                "chunk indexing failed",
                extra={
                    "run_id": run_id,
                    "ticker": ticker.upper() if ticker else None,
                    "indexed_chunks": indexed_chunks,
                    "skipped_chunks": skipped_chunks,
                    "embedding_model": self.settings.embedding_model_name,
                },
                exc_info=True,
            )
            raise

        return IndexingSummary(
            run_id=run_id,
            ticker=ticker.upper() if ticker else None,
            embedding_model=self.settings.embedding_model_name,
            indexed_chunks=indexed_chunks,
            skipped_chunks=skipped_chunks,
        )

    def _create_run(self, *, ticker: str | None) -> str:
        run_id = str(uuid4())
        row = IndexingRunORM(
            run_id=run_id,
            ticker=ticker.upper() if ticker else None,
            embedding_model=self.settings.embedding_model_name,
            status="running",
            indexed_chunks=0,
            skipped_chunks=0,
            started_at=datetime.now(UTC),
            completed_at=None,
            error_message=None,
        )
        self.session.merge(row)
        self.session.commit()
        return run_id

    def _complete_run(
        self,
        *,
        run_id: str,
        indexed_chunks: int,
        skipped_chunks: int,
        error_message: str | None = None,
    ) -> None:
        if error_message is not None or not self.session.is_active:
            self.session.rollback()
        row = self.session.get(IndexingRunORM, run_id)
        if row is None:
            return
        row.status = "failed" if error_message else "completed"
        row.indexed_chunks = indexed_chunks
        row.skipped_chunks = skipped_chunks
        row.completed_at = datetime.now(UTC)
        row.error_message = error_message
        self.session.commit()

    def _select_chunks(
        self,
        *,
        ticker: str | None,
        limit: int | None,
        force: bool,
    ) -> list[ChunkORM]:
        stmt = select(ChunkORM)
        if ticker:
            stmt = stmt.where(ChunkORM.ticker == ticker.upper())
        if not force:
            subquery = select(ChunkEmbeddingORM.chunk_id).where(
                ChunkEmbeddingORM.embedding_model == self.settings.embedding_model_name
            )
            stmt = stmt.where(ChunkORM.chunk_id.not_in(subquery))
        stmt = stmt.order_by(ChunkORM.published_at.desc().nullslast(), ChunkORM.chunk_id.asc())
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))

    def _count_already_indexed_chunks(self, *, ticker: str | None) -> int:
        stmt = select(ChunkEmbeddingORM).where(
            ChunkEmbeddingORM.embedding_model == self.settings.embedding_model_name
        )
        if ticker:
            stmt = stmt.where(ChunkEmbeddingORM.ticker == ticker.upper())
        return len(list(self.session.scalars(stmt)))

    def _embedding_text_from_chunk(self, row: ChunkORM) -> str:
        return (
            f"{row.title}\n"
            f"document_type={row.document_type} form_type={row.form_type} section={row.section}\n"
            f"speaker_role={row.speaker_role} publisher={row.publisher}\n"
            f"published_at={row.published_at}\n"
            f"{row.content[:4000]}"
        )


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.12g}" for value in values) + "]"

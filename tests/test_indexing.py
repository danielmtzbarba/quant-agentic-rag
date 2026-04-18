from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from stock_agent_rag.db import Base, ChunkEmbeddingORM, ChunkORM, IndexingRunORM
from stock_agent_rag.indexing import ChunkIndexingService


class StubEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


def test_chunk_indexing_service_persists_embeddings_and_run() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    settings = type(
        "StubSettings",
        (),
        {
            "embedding_model_name": "stub-embeddings",
            "retrieval_embedding_batch_size": 8,
        },
    )()

    with Session(engine, expire_on_commit=False) as session:
        session.add(
            ChunkORM(
                chunk_id="chunk-1",
                source_id="chunk-1",
                document_id="doc-1",
                ticker="NVDA",
                title="Chunk title",
                content="Chunk body",
                document_type="news",
                provider="alpha_vantage",
                published_at=datetime(2026, 4, 17, tzinfo=UTC),
                chunk_index=0,
                metadata_version="1.0",
            )
        )
        session.commit()

        service = ChunkIndexingService(
            session,
            settings=settings,
            embedding_provider=StubEmbeddingProvider(),
        )
        summary = service.index_chunks(ticker="NVDA")

        assert summary.indexed_chunks == 1
        assert session.scalar(select(ChunkEmbeddingORM.chunk_id)) == "chunk-1"
        assert session.scalar(select(ChunkEmbeddingORM.embedding_model)) == "stub-embeddings"
        assert session.scalar(select(ChunkEmbeddingORM.embedding_vector)) is not None
        assert session.scalar(select(IndexingRunORM.status)) == "completed"


def test_chunk_indexing_service_skips_existing_embeddings_without_force() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    settings = type(
        "StubSettings",
        (),
        {
            "embedding_model_name": "stub-embeddings",
            "retrieval_embedding_batch_size": 8,
        },
    )()

    with Session(engine, expire_on_commit=False) as session:
        session.add(
            ChunkORM(
                chunk_id="chunk-1",
                source_id="chunk-1",
                document_id="doc-1",
                ticker="NVDA",
                title="Chunk title",
                content="Chunk body",
                document_type="news",
                provider="alpha_vantage",
                published_at=datetime(2026, 4, 17, tzinfo=UTC),
                chunk_index=0,
                metadata_version="1.0",
            )
        )
        session.add(
            ChunkEmbeddingORM(
                chunk_id="chunk-1",
                document_id="doc-1",
                ticker="NVDA",
                embedding_model="stub-embeddings",
                embedding_dimensions=2,
                embedding_json=[1.0, 1.0],
                indexed_at=datetime(2026, 4, 17, tzinfo=UTC),
            )
        )
        session.commit()

        service = ChunkIndexingService(
            session,
            settings=settings,
            embedding_provider=StubEmbeddingProvider(),
        )
        summary = service.index_chunks(ticker="NVDA")

        assert summary.indexed_chunks == 0
        assert summary.skipped_chunks == 1

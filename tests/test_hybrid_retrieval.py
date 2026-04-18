from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from stock_agent_rag.db import Base, ChunkEmbeddingORM, ChunkORM
from stock_agent_rag.retrieval import HeuristicQueryPlanner, HeuristicReranker, HybridRetriever


class StubEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def _vectorize(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if any(token in lowered for token in ("guidance", "outlook", "tone")) else 0.0,
            1.0 if any(token in lowered for token in ("demand", "orders", "enterprise")) else 0.0,
            1.0 if any(token in lowered for token in ("risk", "regulation", "export")) else 0.0,
        ]


def _build_retriever(*, session_factory: sessionmaker[Session]) -> HybridRetriever:
    settings = type(
        "StubSettings",
        (),
        {
            "default_top_k": 4,
            "retrieval_candidate_pool": 12,
            "retrieval_rrf_k": 20,
            "retrieval_neighbor_window": 1,
            "retrieval_neighbor_limit": 2,
            "retrieval_rerank_top_n": 8,
            "retrieval_query_plan_limit": 4,
            "retrieval_max_per_document": 2,
            "retrieval_max_news_chunks": 2,
            "retrieval_max_transcript_chunks": 3,
            "retrieval_max_filing_chunks": 3,
            "retrieval_embedding_batch_size": 8,
            "retrieval_sentiment_recency_days": 45,
            "retrieval_risk_recency_days": 120,
            "embedding_model_name": "stub-embeddings",
            "reranker_model_name": "stub-reranker",
            "openai_api_key": None,
        },
    )()
    return HybridRetriever(
        settings=settings,
        session_factory=session_factory,
        embedding_provider=StubEmbeddingProvider(),
        reranker=HeuristicReranker(),
        query_planner=HeuristicQueryPlanner(settings),
    )


def test_hybrid_retrieval_prefilters_and_includes_neighbors() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    now = datetime.now(UTC)

    with session_factory() as session:
        session.add_all(
            [
                ChunkORM(
                    chunk_id="transcript-0",
                    source_id="transcript-0",
                    document_id="transcript-doc",
                    ticker="NVDA",
                    title="Q1 call",
                    content="Management raised guidance and highlighted enterprise demand.",
                    document_type="transcript",
                    provider="alpha_vantage",
                    published_at=now,
                    chunk_index=0,
                    metadata_version="1.0",
                    speaker="Jensen Huang",
                    speaker_role="Chief Executive Officer",
                ),
                ChunkORM(
                    chunk_id="transcript-1",
                    source_id="transcript-1",
                    document_id="transcript-doc",
                    ticker="NVDA",
                    title="Q1 call follow-up",
                    content="The team described enterprise orders and customer expansion.",
                    document_type="transcript",
                    provider="alpha_vantage",
                    published_at=now,
                    chunk_index=1,
                    metadata_version="1.0",
                    speaker="Colette Kress",
                    speaker_role="Chief Financial Officer",
                ),
                ChunkORM(
                    chunk_id="news-fresh",
                    source_id="news-fresh",
                    document_id="news-doc",
                    ticker="NVDA",
                    title="Reuters on demand",
                    content="Reuters said enterprise demand remains strong.",
                    document_type="news",
                    provider="alpha_vantage",
                    publisher="Reuters",
                    published_at=now - timedelta(days=2),
                    chunk_index=0,
                    metadata_version="1.0",
                ),
                ChunkORM(
                    chunk_id="news-stale",
                    source_id="news-stale",
                    document_id="news-old-doc",
                    ticker="NVDA",
                    title="Old article",
                    content="Older sentiment article.",
                    document_type="news",
                    provider="alpha_vantage",
                    publisher="Reuters",
                    published_at=now - timedelta(days=400),
                    chunk_index=0,
                    metadata_version="1.0",
                ),
                ChunkORM(
                    chunk_id="filing-risk",
                    source_id="filing-risk",
                    document_id="filing-doc",
                    ticker="NVDA",
                    title="Risk factors",
                    content="Regulatory risk remains elevated.",
                    document_type="filing",
                    provider="sec",
                    form_type="10-K",
                    section="item_1a_risk_factors",
                    published_at=now - timedelta(days=15),
                    chunk_index=0,
                    metadata_version="1.0",
                ),
                ChunkEmbeddingORM(
                    chunk_id="transcript-0",
                    document_id="transcript-doc",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[1.0, 1.0, 0.0],
                    indexed_at=now,
                ),
                ChunkEmbeddingORM(
                    chunk_id="transcript-1",
                    document_id="transcript-doc",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[0.0, 1.0, 0.0],
                    indexed_at=now,
                ),
                ChunkEmbeddingORM(
                    chunk_id="news-fresh",
                    document_id="news-doc",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[0.0, 1.0, 0.0],
                    indexed_at=now,
                ),
                ChunkEmbeddingORM(
                    chunk_id="news-stale",
                    document_id="news-old-doc",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[1.0, 0.0, 0.0],
                    indexed_at=now,
                ),
                ChunkEmbeddingORM(
                    chunk_id="filing-risk",
                    document_id="filing-doc",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[0.0, 0.0, 1.0],
                    indexed_at=now,
                ),
            ]
        )
        session.commit()

    retriever = _build_retriever(session_factory=session_factory)
    results = retriever.search(
        query="management guidance and enterprise orders",
        ticker="NVDA",
        top_k=2,
        profile="sentiment",
    )

    source_ids = [record.source_id for record in results]
    assert "transcript-0" in source_ids
    assert "transcript-1" in source_ids
    assert "news-stale" not in source_ids
    assert "filing-risk" not in source_ids


def test_hybrid_retrieval_uses_semantic_candidates_in_fusion() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    now = datetime.now(UTC)

    with session_factory() as session:
        session.add_all(
            [
                ChunkORM(
                    chunk_id="lexical-hit",
                    source_id="lexical-hit",
                    document_id="doc-1",
                    ticker="NVDA",
                    title="Management guidance",
                    content="Guidance improved with modest lexical overlap.",
                    document_type="transcript",
                    provider="alpha_vantage",
                    published_at=now,
                    chunk_index=0,
                    metadata_version="1.0",
                    speaker_role="Chief Executive Officer",
                ),
                ChunkORM(
                    chunk_id="semantic-hit",
                    source_id="semantic-hit",
                    document_id="doc-2",
                    ticker="NVDA",
                    title="Raised outlook",
                    content="The company described stronger customer orders and tone.",
                    document_type="news",
                    provider="alpha_vantage",
                    publisher="Reuters",
                    published_at=now,
                    chunk_index=0,
                    metadata_version="1.0",
                ),
                ChunkEmbeddingORM(
                    chunk_id="lexical-hit",
                    document_id="doc-1",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[1.0, 0.0, 0.0],
                    indexed_at=now,
                ),
                ChunkEmbeddingORM(
                    chunk_id="semantic-hit",
                    document_id="doc-2",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[1.0, 1.0, 0.0],
                    indexed_at=now,
                ),
            ]
        )
        session.commit()

    retriever = _build_retriever(session_factory=session_factory)
    results = retriever.search(
        query="guidance and enterprise demand",
        ticker="NVDA",
        top_k=2,
        profile="sentiment",
    )

    source_ids = [record.source_id for record in results]
    assert "lexical-hit" in source_ids
    assert "semantic-hit" in source_ids


def test_hybrid_retrieval_applies_source_diversity_controls() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    now = datetime.now(UTC)

    with session_factory() as session:
        session.add_all(
            [
                ChunkORM(
                    chunk_id=f"news-{idx}",
                    source_id=f"news-{idx}",
                    document_id=f"news-doc-{idx}",
                    ticker="NVDA",
                    title=f"News {idx}",
                    content="Guidance demand tone",
                    document_type="news",
                    provider="alpha_vantage",
                    publisher="Reuters",
                    published_at=now - timedelta(hours=idx),
                    chunk_index=0,
                    metadata_version="1.0",
                )
                for idx in range(4)
            ]
            + [
                ChunkORM(
                    chunk_id="transcript-main",
                    source_id="transcript-main",
                    document_id="transcript-doc",
                    ticker="NVDA",
                    title="Transcript",
                    content="Management guidance remained constructive.",
                    document_type="transcript",
                    provider="alpha_vantage",
                    published_at=now,
                    chunk_index=0,
                    metadata_version="1.0",
                    speaker_role="Chief Executive Officer",
                )
            ]
            + [
                ChunkEmbeddingORM(
                    chunk_id=f"news-{idx}",
                    document_id=f"news-doc-{idx}",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[1.0, 1.0, 0.0],
                    indexed_at=now,
                )
                for idx in range(4)
            ]
            + [
                ChunkEmbeddingORM(
                    chunk_id="transcript-main",
                    document_id="transcript-doc",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[1.0, 1.0, 0.0],
                    indexed_at=now,
                )
            ]
        )
        session.commit()

    retriever = _build_retriever(session_factory=session_factory)
    results = retriever.search(
        query="guidance and demand outlook",
        ticker="NVDA",
        top_k=3,
        profile="sentiment",
    )

    doc_types = [record.document_type for record in results]
    assert "transcript" in doc_types
    assert doc_types.count("news") <= 2


def test_hybrid_retrieval_prefers_latest_transcript_quarter() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    now = datetime.now(UTC)

    with session_factory() as session:
        session.add_all(
            [
                ChunkORM(
                    chunk_id="transcript-latest",
                    source_id="transcript-latest",
                    document_id="transcript-latest-doc",
                    ticker="NVDA",
                    title="Latest call",
                    content="Management guidance was constructive and demand was strong.",
                    document_type="transcript",
                    provider="alpha_vantage",
                    published_at=now - timedelta(days=10),
                    chunk_index=0,
                    metadata_version="1.0",
                    speaker_role="Chief Executive Officer",
                ),
                ChunkORM(
                    chunk_id="transcript-old",
                    source_id="transcript-old",
                    document_id="transcript-old-doc",
                    ticker="NVDA",
                    title="Old call",
                    content="Management guidance was constructive and demand was strong.",
                    document_type="transcript",
                    provider="alpha_vantage",
                    published_at=now - timedelta(days=170),
                    chunk_index=0,
                    metadata_version="1.0",
                    speaker_role="Chief Executive Officer",
                ),
                ChunkEmbeddingORM(
                    chunk_id="transcript-latest",
                    document_id="transcript-latest-doc",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[1.0, 1.0, 0.0],
                    indexed_at=now,
                ),
                ChunkEmbeddingORM(
                    chunk_id="transcript-old",
                    document_id="transcript-old-doc",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[1.0, 1.0, 0.0],
                    indexed_at=now,
                ),
            ]
        )
        session.commit()

    retriever = _build_retriever(session_factory=session_factory)
    results = retriever.search(
        query="guidance and demand outlook",
        ticker="NVDA",
        top_k=1,
        profile="sentiment",
    )

    assert results[0].source_id == "transcript-latest"


def test_hybrid_retrieval_prefers_latest_filing_by_form_type() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    now = datetime.now(UTC)

    with session_factory() as session:
        session.add_all(
            [
                ChunkORM(
                    chunk_id="filing-latest-10q",
                    source_id="filing-latest-10q",
                    document_id="filing-latest-10q-doc",
                    ticker="NVDA",
                    title="Latest 10-Q",
                    content="Revenue growth and margins remained strong.",
                    document_type="filing",
                    provider="sec",
                    form_type="10-Q",
                    section="part_i_item_2_mda",
                    published_at=now - timedelta(days=20),
                    chunk_index=0,
                    metadata_version="1.0",
                ),
                ChunkORM(
                    chunk_id="filing-old-10q",
                    source_id="filing-old-10q",
                    document_id="filing-old-10q-doc",
                    ticker="NVDA",
                    title="Old 10-Q",
                    content="Revenue growth and margins remained strong.",
                    document_type="filing",
                    provider="sec",
                    form_type="10-Q",
                    section="part_i_item_2_mda",
                    published_at=now - timedelta(days=180),
                    chunk_index=0,
                    metadata_version="1.0",
                ),
                ChunkEmbeddingORM(
                    chunk_id="filing-latest-10q",
                    document_id="filing-latest-10q-doc",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[0.0, 1.0, 0.0],
                    indexed_at=now,
                ),
                ChunkEmbeddingORM(
                    chunk_id="filing-old-10q",
                    document_id="filing-old-10q-doc",
                    ticker="NVDA",
                    embedding_model="stub-embeddings",
                    embedding_dimensions=3,
                    embedding_json=[0.0, 1.0, 0.0],
                    indexed_at=now,
                ),
            ]
        )
        session.commit()

    retriever = _build_retriever(session_factory=session_factory)
    results = retriever.search(
        query="revenue growth and margins",
        ticker="NVDA",
        top_k=1,
        profile="fundamentals",
    )

    assert results[0].source_id == "filing-latest-10q"

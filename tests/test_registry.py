from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from stock_agent_rag.db import Base, ChunkORM, DocumentORM, IngestionRunORM, SourceRegistryORM
from stock_agent_rag.registry import RegistryService
from stock_agent_rag.schemas import DocumentRecord, EvidenceChunk, FilingSection


def test_registry_persists_document_chunk_and_run() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        service = RegistryService(session)
        run_id = service.create_ingestion_run(
            source_type="filing",
            ticker="AAPL",
            form_type="10-K",
            metadata_version="1.0",
        )

        document = DocumentRecord(
            document_id="aapl-10-k-test",
            source_type="filing",
            ticker="AAPL",
            title="AAPL 10-K filing",
            provider="sec-edgar-downloader",
            form_type="10-K",
            published_at=datetime(2024, 1, 31, tzinfo=UTC),
            raw_checksum="abc123",
            raw_path="/tmp/raw.txt",
            cleaned_text="cleaned text",
            sections=[
                FilingSection(
                    section_id="item_7_mda",
                    item_label="Item 7",
                    title="Management Discussion",
                    content="section text",
                    start_offset=0,
                    end_offset=12,
                )
            ],
        )
        chunk = EvidenceChunk(
            chunk_id="chunk-1",
            source_id="chunk-1",
            document_id=document.document_id,
            ticker="AAPL",
            title="AAPL 10-K filing | Management Discussion",
            content="chunk text",
            document_type="filing",
            provider=document.provider,
            form_type=document.form_type,
            section="item_7_mda",
            chunk_index=0,
            metadata_version="1.0",
            speaker="Tim Cook",
            speaker_role="Chief Executive Officer",
        )

        service.upsert_document(document=document, normalized_path="/tmp/document.json")
        service.upsert_chunks(chunks=[chunk], chunk_path="/tmp/chunks.jsonl")
        service.complete_ingestion_run(
            run_id=run_id,
            processed_documents=1,
            chunk_count=1,
        )

        assert session.scalar(select(DocumentORM.document_id)) == "aapl-10-k-test"
        assert session.scalar(select(ChunkORM.chunk_id)) == "chunk-1"
        assert session.scalar(select(ChunkORM.speaker)) == "Tim Cook"
        assert session.scalar(select(SourceRegistryORM.latest_document_id)) == "aapl-10-k-test"
        assert session.scalar(select(IngestionRunORM.status)) == "completed"


def test_complete_ingestion_run_recovers_from_failed_session_state() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        service = RegistryService(session)
        run_id = service.create_ingestion_run(
            source_type="transcript",
            ticker="NVDA",
            form_type="Q1-2026",
            metadata_version="1.0",
        )

        original_commit = session.commit
        commit_calls = {"count": 0}

        def flaky_commit() -> None:
            commit_calls["count"] += 1
            if commit_calls["count"] == 1:
                raise RuntimeError("boom")
            original_commit()

        session.commit = flaky_commit  # type: ignore[method-assign]

        with pytest.raises(RuntimeError):
            service.upsert_chunks(
                chunks=[
                    EvidenceChunk(
                        chunk_id="chunk-1",
                        source_id="chunk-1",
                        document_id="doc-1",
                        ticker="NVDA",
                        title="Test chunk",
                        content="hello",
                        document_type="transcript",
                        provider="alpha_vantage",
                        chunk_index=0,
                    )
                ],
                chunk_path="/tmp/chunks.jsonl",
            )

        session.commit = original_commit  # type: ignore[method-assign]
        service.complete_ingestion_run(
            run_id=run_id,
            processed_documents=0,
            chunk_count=0,
            error_message="boom",
        )

        assert session.scalar(select(IngestionRunORM.status)) == "failed"

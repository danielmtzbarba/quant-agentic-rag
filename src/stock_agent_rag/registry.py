from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from .db import ChunkORM, DocumentORM, IngestionRunORM, SourceRegistryORM
from .schemas import DocumentRecord, EvidenceChunk


class RegistryService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _commit(self) -> None:
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

    def create_ingestion_run(
        self,
        *,
        source_type: str,
        ticker: str,
        form_type: str | None,
        metadata_version: str,
    ) -> str:
        run_id = str(uuid4())
        run = IngestionRunORM(
            run_id=run_id,
            source_type=source_type,
            ticker=ticker,
            form_type=form_type,
            status="running",
            metadata_version=metadata_version,
            processed_documents=0,
            chunk_count=0,
            started_at=datetime.now(UTC),
            completed_at=None,
            error_message=None,
        )
        self.session.merge(run)
        self._commit()
        return run_id

    def complete_ingestion_run(
        self,
        *,
        run_id: str,
        processed_documents: int,
        chunk_count: int,
        error_message: str | None = None,
    ) -> None:
        if error_message is not None or not self.session.is_active:
            self.session.rollback()
        run = self.session.get(IngestionRunORM, run_id)
        if run is None:
            return
        run.processed_documents = processed_documents
        run.chunk_count = chunk_count
        run.completed_at = datetime.now(UTC)
        run.error_message = error_message
        run.status = "failed" if error_message else "completed"
        self._commit()

    def upsert_document(
        self,
        *,
        document: DocumentRecord,
        normalized_path: str,
    ) -> None:
        row = DocumentORM(
            document_id=document.document_id,
            source_type=document.source_type,
            ticker=document.ticker,
            title=document.title,
            provider=document.provider,
            form_type=document.form_type,
            published_at=document.published_at,
            as_of_date=document.as_of_date,
            source_url=document.source_url,
            accession_number=document.accession_number,
            cik=document.cik,
            metadata_version=document.metadata_version,
            raw_checksum=document.raw_checksum,
            raw_path=document.raw_path,
            normalized_path=normalized_path,
            cleaned_text=document.cleaned_text,
            publisher=document.publisher,
            sentiment_label=document.sentiment_label,
            sentiment_score=document.sentiment_score,
            ticker_relevance_score=document.ticker_relevance_score,
            entity_title_match=document.entity_title_match,
            entity_body_match=document.entity_body_match,
            news_relevance_score=document.news_relevance_score,
            news_relevance_tier=document.news_relevance_tier,
            source_quality_tier=document.source_quality_tier,
            section_count=len(document.sections),
            sections_json=[section.model_dump(mode="json") for section in document.sections],
            transcript_turn_count=len(document.transcript_turns),
            transcript_turns_json=[
                turn.model_dump(mode="json") for turn in document.transcript_turns
            ],
        )
        self.session.merge(row)

        source_key = (
            f"{document.source_type}:{document.ticker}:"
            f"{document.form_type or 'unknown'}:{document.provider}"
        )
        registry = SourceRegistryORM(
            source_key=source_key,
            source_type=document.source_type,
            ticker=document.ticker,
            provider=document.provider,
            latest_document_id=document.document_id,
            latest_published_at=document.published_at,
            metadata_version=document.metadata_version,
            active=True,
        )
        self.session.merge(registry)
        self._commit()

    def upsert_chunks(self, *, chunks: list[EvidenceChunk], chunk_path: str) -> None:
        for chunk in chunks:
            row = ChunkORM(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                document_id=chunk.document_id,
                ticker=chunk.ticker,
                title=chunk.title,
                content=chunk.content,
                document_type=chunk.document_type,
                provider=chunk.provider,
                form_type=chunk.form_type,
                section=chunk.section,
                source_url=chunk.source_url,
                published_at=chunk.published_at,
                accession_number=chunk.accession_number,
                chunk_index=chunk.chunk_index,
                metadata_version=chunk.metadata_version,
                score=chunk.score,
                chunk_path=chunk_path,
                speaker=chunk.speaker,
                speaker_role=chunk.speaker_role,
                publisher=chunk.publisher,
                sentiment_label=chunk.sentiment_label,
                sentiment_score=chunk.sentiment_score,
                ticker_relevance_score=chunk.ticker_relevance_score,
                entity_title_match=chunk.entity_title_match,
                entity_body_match=chunk.entity_body_match,
                news_relevance_score=chunk.news_relevance_score,
                news_relevance_tier=chunk.news_relevance_tier,
                source_quality_tier=chunk.source_quality_tier,
            )
            self.session.merge(row)
        self._commit()

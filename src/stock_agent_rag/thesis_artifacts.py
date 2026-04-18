from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from .artifact_storage import LocalArtifactStorage, ObjectStorageProvider, S3ArtifactStorage
from .config import Settings, get_settings
from .db import ThesisArtifactORM
from .schemas import ContradictionRecord, ThesisArtifactSummary


class ThesisArtifactService:
    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        storage_provider: ObjectStorageProvider | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.storage_provider = storage_provider or self._build_storage_provider()

    def persist(
        self,
        *,
        run_id: str,
        ticker: str,
        question: str,
        result: dict[str, object],
    ) -> ThesisArtifactSummary:
        created_at = datetime.now(UTC)
        thesis_id = str(uuid4())
        content = self._render_markdown(
            run_id=run_id,
            ticker=ticker,
            question=question,
            result=result,
            created_at=created_at,
        )
        object_key = self._build_object_key(ticker=ticker, run_id=run_id, created_at=created_at)
        content_type = "text/markdown; charset=utf-8"
        stored = self.storage_provider.put_text(
            bucket=self.settings.thesis_artifact_bucket,
            object_key=object_key,
            content=content,
            content_type=content_type,
        )
        if self.settings.thesis_artifact_local_mirror and stored.storage_provider != "local":
            mirror = LocalArtifactStorage(self.settings).put_text(
                bucket=self.settings.thesis_artifact_bucket,
                object_key=object_key,
                content=content,
                content_type=content_type,
            )
            markdown_path = mirror.markdown_path
        else:
            markdown_path = stored.markdown_path

        markdown_checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        thesis_hash = hashlib.sha256(
            str(result.get("report", "")).encode("utf-8")
        ).hexdigest()
        verification_metrics = result.get("verification_metrics")
        retrieval_metrics = result.get("retrieval_metrics")
        runtime_metrics = result.get("runtime_metrics")
        contradictions = self._dump_contradictions(result.get("contradictions"))
        retrieved_source_ids = self._source_ids(result.get("retrieved_evidence"))
        row = ThesisArtifactORM(
            thesis_id=thesis_id,
            run_id=run_id,
            ticker=ticker,
            question=question,
            artifact_version="1.0",
            created_at=created_at,
            storage_provider=stored.storage_provider,
            bucket=stored.bucket,
            object_key=stored.object_key,
            content_type=stored.content_type,
            markdown_path=markdown_path,
            markdown_checksum=markdown_checksum,
            object_etag=stored.etag,
            status="completed",
            verification_status=str(result.get("verification_status") or "unknown"),
            deterministic_verifier_status=(
                str(verification_metrics.get("deterministic_status"))
                if isinstance(verification_metrics, dict)
                and verification_metrics.get("deterministic_status") is not None
                else None
            ),
            model_name=self._model_name(result.get("model_metadata")),
            embedding_model=self.settings.embedding_model_name,
            retrieved_source_count=len(retrieved_source_ids),
            cited_source_count=(
                int(verification_metrics.get("cited_retrieved_sources", 0))
                if isinstance(verification_metrics, dict)
                else 0
            ),
            citation_coverage=(
                float(verification_metrics.get("citation_coverage"))
                if isinstance(verification_metrics, dict)
                and verification_metrics.get("citation_coverage") is not None
                else None
            ),
            structured_findings_count=self._structured_findings_count(result),
            unsupported_findings_count=(
                int(verification_metrics.get("unsupported_findings", 0))
                if isinstance(verification_metrics, dict)
                else 0
            ),
            partially_grounded_findings_count=(
                int(verification_metrics.get("partially_grounded_findings", 0))
                if isinstance(verification_metrics, dict)
                else 0
            ),
            contradictions_count=len(contradictions),
            latency_ms=(
                float(result.get("latency_ms"))
                if isinstance(result.get("latency_ms"), int | float)
                else None
            ),
            estimated_cost_usd=(
                float(result.get("estimated_cost_usd"))
                if isinstance(result.get("estimated_cost_usd"), int | float)
                else None
            ),
            thesis_word_count=len(str(result.get("report", "")).split()),
            thesis_hash=thesis_hash,
            top_source_ids_json=retrieved_source_ids[:10],
            contradictions_json=contradictions,
            retrieval_metrics_json=(
                retrieval_metrics if isinstance(retrieval_metrics, dict) else None
            ),
            verification_metrics_json=(
                verification_metrics if isinstance(verification_metrics, dict) else None
            ),
            runtime_metrics_json=runtime_metrics if isinstance(runtime_metrics, dict) else None,
            tags_json=[],
        )
        self.session.add(row)
        self.session.commit()
        return ThesisArtifactSummary(
            thesis_id=thesis_id,
            run_id=run_id,
            ticker=ticker,
            storage_provider=stored.storage_provider,
            bucket=stored.bucket,
            object_key=stored.object_key,
            markdown_path=markdown_path,
            markdown_checksum=markdown_checksum,
            thesis_hash=thesis_hash,
        )

    def _build_storage_provider(self) -> ObjectStorageProvider:
        provider = self.settings.thesis_storage_provider.lower().strip()
        if provider == "local":
            return LocalArtifactStorage(self.settings)
        if provider == "s3":
            return S3ArtifactStorage(self.settings)
        raise RuntimeError(
            f"Unsupported THESIS_STORAGE_PROVIDER={self.settings.thesis_storage_provider}"
        )

    def _build_object_key(self, *, ticker: str, run_id: str, created_at: datetime) -> str:
        return (
            f"theses/{ticker.upper()}/{created_at:%Y}/{created_at:%m}/{run_id}.md"
        )

    def _render_markdown(
        self,
        *,
        run_id: str,
        ticker: str,
        question: str,
        result: dict[str, object],
        created_at: datetime,
    ) -> str:
        model_name = self._model_name(result.get("model_metadata")) or "unknown"
        return (
            f"# {ticker} Research Thesis\n\n"
            f"- Run ID: `{run_id}`\n"
            f"- Ticker: `{ticker}`\n"
            f"- Question: {question}\n"
            f"- Generated At: `{created_at.isoformat()}`\n"
            f"- Verification Status: `{result.get('verification_status', 'unknown')}`\n"
            f"- Model: `{model_name}`\n\n"
            f"## Plan\n\n{result.get('plan', '')}\n\n"
            f"## Thesis\n\n{result.get('report', '')}\n\n"
            f"## Verification Summary\n\n{result.get('verification_summary', '')}\n\n"
            f"## Contradictions\n\n{result.get('contradiction_summary', 'None.')}\n"
        )

    def _model_name(self, metadata: object) -> str | None:
        if not isinstance(metadata, dict):
            return None
        models = metadata.get("models")
        if isinstance(models, list) and models:
            first = models[0]
            if isinstance(first, str):
                return first
        return None

    def _structured_findings_count(self, result: dict[str, object]) -> int:
        verification_metrics = result.get("verification_metrics")
        if isinstance(verification_metrics, dict):
            value = verification_metrics.get("structured_findings")
            if isinstance(value, int):
                return value
        total = 0
        for key in ("fundamentals_analysis", "sentiment_analysis", "risk_analysis"):
            analysis = result.get(key)
            findings = getattr(analysis, "findings", None)
            if isinstance(findings, list):
                total += len(findings)
        return total

    def _source_ids(self, records: object) -> list[str]:
        if not isinstance(records, list):
            return []
        return [
            record.source_id
            for record in records
            if hasattr(record, "source_id") and isinstance(record.source_id, str)
        ]

    def _dump_contradictions(self, contradictions: object) -> list[dict]:
        if not isinstance(contradictions, list):
            return []
        dumped: list[dict] = []
        for item in contradictions:
            if isinstance(item, ContradictionRecord):
                dumped.append(item.model_dump(mode="json"))
        return dumped

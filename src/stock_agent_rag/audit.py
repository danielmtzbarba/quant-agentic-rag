from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from .db import ResearchRunORM
from .schemas import AnalystOutput, ThesisPreparation


class ResearchAuditService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _commit(self) -> None:
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

    def create_research_run(self, *, ticker: str, question: str) -> str:
        run_id = str(uuid4())
        row = ResearchRunORM(
            run_id=run_id,
            ticker=ticker,
            question=question,
            status="running",
            verification_status=None,
            started_at=datetime.now(UTC),
            completed_at=None,
            latency_ms=None,
            plan=None,
            report=None,
            verification_summary=None,
            retrieved_source_ids_json=[],
            node_metrics_json=None,
            token_usage_json=None,
            model_metadata_json=None,
            runtime_metrics_json=None,
            retrieval_metrics_json=None,
            estimated_cost_usd=None,
            fundamentals_analysis_json=None,
            sentiment_analysis_json=None,
            risk_analysis_json=None,
            thesis_preparation_json=None,
            verification_metrics_json=None,
            error_message=None,
        )
        self.session.merge(row)
        self._commit()
        return run_id

    def complete_research_run(
        self,
        *,
        run_id: str,
        ticker: str,
        question: str,
        latency_ms: float | None,
        result: dict[str, object] | None = None,
        error_message: str | None = None,
    ) -> None:
        if error_message is not None or not self.session.is_active:
            self.session.rollback()

        row = self.session.get(ResearchRunORM, run_id)
        if row is None:
            row = ResearchRunORM(
                run_id=run_id,
                ticker=ticker,
                question=question,
                status="running",
                verification_status=None,
                started_at=datetime.now(UTC),
                completed_at=None,
                latency_ms=None,
                plan=None,
                report=None,
                verification_summary=None,
                retrieved_source_ids_json=[],
                node_metrics_json=None,
                token_usage_json=None,
                model_metadata_json=None,
                runtime_metrics_json=None,
                retrieval_metrics_json=None,
                estimated_cost_usd=None,
                fundamentals_analysis_json=None,
                sentiment_analysis_json=None,
                risk_analysis_json=None,
                thesis_preparation_json=None,
                verification_metrics_json=None,
                error_message=None,
            )
            self.session.add(row)

        row.status = "failed" if error_message else "completed"
        row.verification_status = (
            None if result is None else str(result.get("verification_status") or "unknown")
        )
        row.completed_at = datetime.now(UTC)
        row.latency_ms = latency_ms
        row.error_message = error_message

        if result is not None:
            row.plan = str(result.get("plan") or "")
            row.report = str(result.get("report") or "")
            row.verification_summary = str(result.get("verification_summary") or "")
            retrieved_evidence = result.get("retrieved_evidence") or []
            row.retrieved_source_ids_json = [
                record.source_id for record in retrieved_evidence if hasattr(record, "source_id")
            ]
            node_metrics = result.get("node_metrics")
            row.node_metrics_json = node_metrics if isinstance(node_metrics, dict) else None
            token_usage = result.get("token_usage")
            row.token_usage_json = token_usage if isinstance(token_usage, dict) else None
            model_metadata = result.get("model_metadata")
            row.model_metadata_json = model_metadata if isinstance(model_metadata, dict) else None
            runtime_metrics = result.get("runtime_metrics")
            row.runtime_metrics_json = (
                runtime_metrics if isinstance(runtime_metrics, dict) else None
            )
            retrieval_metrics = result.get("retrieval_metrics")
            row.retrieval_metrics_json = (
                retrieval_metrics if isinstance(retrieval_metrics, dict) else None
            )
            estimated_cost = result.get("estimated_cost_usd")
            row.estimated_cost_usd = (
                float(estimated_cost) if isinstance(estimated_cost, int | float) else None
            )
            row.fundamentals_analysis_json = self._analysis_dump(
                result.get("fundamentals_analysis")
            )
            row.sentiment_analysis_json = self._analysis_dump(result.get("sentiment_analysis"))
            row.risk_analysis_json = self._analysis_dump(result.get("risk_analysis"))
            row.thesis_preparation_json = self._thesis_preparation_dump(
                result.get("thesis_preparation")
            )
            verification_metrics = result.get("verification_metrics")
            row.verification_metrics_json = (
                verification_metrics if isinstance(verification_metrics, dict) else None
            )

        self._commit()

    def _analysis_dump(self, value: object) -> dict | None:
        if isinstance(value, AnalystOutput):
            return value.model_dump(mode="json")
        return None

    def _thesis_preparation_dump(self, value: object) -> dict | None:
        if isinstance(value, ThesisPreparation):
            return value.model_dump(mode="json")
        return None

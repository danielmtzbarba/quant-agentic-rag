from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Any

from .audit import ResearchAuditService
from .config import get_settings
from .db import get_db_session, initialize_database
from .logging import get_logger
from .schemas import ResearchRequest, ResearchResponse
from .telemetry import (
    aggregate_estimated_cost_usd,
    aggregate_runtime_metrics,
    aggregate_token_usage,
    build_retrieval_metrics,
    collect_model_metadata,
)
from .thesis_artifacts import ThesisArtifactService
from .workflow import build_app

logger = get_logger(__name__)


class ResearchService:
    def __init__(self, app: Any | None = None) -> None:
        self._app = app or build_app()

    def run(self, request: ResearchRequest) -> ResearchResponse:
        settings = get_settings()
        audit_service: ResearchAuditService | None = None
        session = None
        run_id: str | None = None
        ticker = request.ticker.upper()

        if settings.db_enabled:
            initialize_database(settings)
            session = get_db_session()
            audit_service = ResearchAuditService(session)
            run_id = audit_service.create_research_run(ticker=ticker, question=request.question)

        start = perf_counter()
        try:
            result = self._app.invoke({"ticker": ticker, "question": request.question})
            latency_ms = round((perf_counter() - start) * 1000, 2)
            node_metrics = result.get("node_metrics", {})
            if not isinstance(node_metrics, dict):
                node_metrics = {}
            token_usage = aggregate_token_usage(node_metrics)
            model_metadata = collect_model_metadata(node_metrics)
            runtime_metrics = aggregate_runtime_metrics(node_metrics)
            estimated_cost_usd = aggregate_estimated_cost_usd(node_metrics)
            retrieval_metrics = build_retrieval_metrics(
                fundamentals_evidence=result.get("fundamentals_evidence", []),
                sentiment_evidence=result.get("sentiment_evidence", []),
                risk_evidence=result.get("risk_evidence", []),
                retrieved_evidence=result.get("retrieved_evidence", []),
                default_top_k=int(getattr(settings, "default_top_k", 4)),
            )
            result["token_usage"] = token_usage
            result["model_metadata"] = model_metadata
            result["runtime_metrics"] = runtime_metrics
            result["estimated_cost_usd"] = estimated_cost_usd
            result["retrieval_metrics"] = retrieval_metrics
            result["latency_ms"] = latency_ms

            thesis_artifact = None
            if session is not None and run_id is not None:
                try:
                    thesis_artifact = ThesisArtifactService(session).persist(
                        run_id=run_id,
                        ticker=ticker,
                        question=request.question,
                        result=result,
                    )
                    result["thesis_id"] = thesis_artifact.thesis_id
                    result["thesis_storage_provider"] = thesis_artifact.storage_provider
                    result["thesis_bucket"] = thesis_artifact.bucket
                    result["thesis_object_key"] = thesis_artifact.object_key
                    result["thesis_markdown_path"] = thesis_artifact.markdown_path
                except Exception:
                    session.rollback()
                    logger.warning(
                        "thesis artifact persistence failed",
                        extra={"ticker": ticker, "run_id": run_id},
                        exc_info=True,
                    )

            if audit_service is not None and run_id is not None:
                audit_service.complete_research_run(
                    run_id=run_id,
                    ticker=ticker,
                    question=request.question,
                    latency_ms=latency_ms,
                    result=result,
                )
        except Exception as exc:
            latency_ms = round((perf_counter() - start) * 1000, 2)
            if audit_service is not None and run_id is not None:
                audit_service.complete_research_run(
                    run_id=run_id,
                    ticker=ticker,
                    question=request.question,
                    latency_ms=latency_ms,
                    error_message=str(exc),
                )
            if session is not None:
                session.close()
            raise

        logger.info(
            "research workflow completed",
            extra={
                "ticker": ticker,
                "latency_ms": latency_ms,
                "retrieved_sources": len(result.get("retrieved_evidence", [])),
                "verification_status": result.get("verification_status", "unknown"),
                "total_tokens": result.get("token_usage", {}).get("total_tokens", 0),
                "estimated_cost_usd": result.get("estimated_cost_usd"),
            },
        )

        response = ResearchResponse(
            ticker=ticker,
            question=request.question,
            plan=result.get("plan", ""),
            report=result.get("report", ""),
            verification_status=result.get("verification_status", "unknown"),
            verification_summary=result.get("verification_summary", ""),
            retrieved_sources=[record.source_id for record in result.get("retrieved_evidence", [])],
            token_usage=result.get("token_usage", {}),
            model_metadata=result.get("model_metadata", {}),
            runtime_metrics=result.get("runtime_metrics", {}),
            retrieval_metrics=result.get("retrieval_metrics", {}),
            estimated_cost_usd=result.get("estimated_cost_usd"),
            thesis_id=result.get("thesis_id"),
            thesis_storage_provider=result.get("thesis_storage_provider"),
            thesis_bucket=result.get("thesis_bucket"),
            thesis_object_key=result.get("thesis_object_key"),
            thesis_markdown_path=result.get("thesis_markdown_path"),
            latency_ms=latency_ms,
        )
        if session is not None:
            session.close()
        return response


@lru_cache(maxsize=1)
def get_research_service() -> ResearchService:
    return ResearchService()

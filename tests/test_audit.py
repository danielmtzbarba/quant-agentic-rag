from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from stock_agent_rag.audit import ResearchAuditService
from stock_agent_rag.db import Base, ResearchRunORM
from stock_agent_rag.schemas import (
    AnalystFinding,
    AnalystOutput,
    ThesisFinding,
    ThesisPreparation,
    ThesisSectionInput,
)


def test_research_audit_service_persists_structured_artifacts() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        service = ResearchAuditService(session)
        run_id = service.create_research_run(
            ticker="NVDA",
            question="What is the thesis?",
        )

        service.complete_research_run(
            run_id=run_id,
            ticker="NVDA",
            question="What is the thesis?",
            latency_ms=42.0,
            result={
                "plan": "plan",
                "report": "report [source:a1]",
                "verification_status": "fail",
                "verification_summary": "fail summary",
                "retrieved_evidence": [type("Record", (), {"source_id": "a1"})()],
                "node_metrics": {
                    "planner": {
                        "model_name": "gpt-test",
                        "provider": "openai",
                        "temperature": 0.0,
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "retry_count": 1,
                        "timeout_count": 0,
                        "estimated_cost_usd": 0.000123,
                        "latency_ms": 12.0,
                    }
                },
                "token_usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                "model_metadata": {
                    "models": ["gpt-test"],
                    "providers": ["openai"],
                    "temperatures": [0.0],
                },
                "runtime_metrics": {
                    "retry_count": 1,
                    "timeout_count": 0,
                },
                "retrieval_metrics": {
                    "merged_retrieved_count": 1,
                    "merged_hit_rate": 0.25,
                },
                "estimated_cost_usd": 0.000123,
                "fundamentals_analysis": AnalystOutput(
                    summary="summary",
                    findings=[
                        AnalystFinding(
                            finding="Revenue growth is strong",
                            evidence_ids=["a1"],
                            confidence=0.9,
                            missing_data=[],
                        )
                    ],
                ),
                "thesis_preparation": ThesisPreparation(
                    sections=[
                        ThesisSectionInput(
                            section_id="bull_case",
                            title="Bull Case",
                            objective="objective",
                            findings=[
                                ThesisFinding(
                                    analyst="fundamentals",
                                    finding="Revenue growth is strong",
                                    evidence_ids=["a1"],
                                    confidence=0.9,
                                    missing_data=[],
                                )
                            ],
                            evidence_ids=["a1"],
                        )
                    ]
                ),
                "verification_metrics": {
                    "deterministic_status": "fail",
                    "unsupported_findings": 1,
                },
            },
        )

        row = session.scalar(select(ResearchRunORM).where(ResearchRunORM.run_id == run_id))
        assert row is not None
        assert row.status == "completed"
        assert row.verification_status == "fail"
        assert row.retrieved_source_ids_json == ["a1"]
        assert row.node_metrics_json is not None
        assert row.token_usage_json == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        assert row.model_metadata_json == {
            "models": ["gpt-test"],
            "providers": ["openai"],
            "temperatures": [0.0],
        }
        assert row.runtime_metrics_json == {
            "retry_count": 1,
            "timeout_count": 0,
        }
        assert row.retrieval_metrics_json == {
            "merged_retrieved_count": 1,
            "merged_hit_rate": 0.25,
        }
        assert row.estimated_cost_usd == 0.000123
        assert row.fundamentals_analysis_json is not None
        assert row.thesis_preparation_json is not None
        assert row.verification_metrics_json == {
            "deterministic_status": "fail",
            "unsupported_findings": 1,
        }


def test_research_audit_service_marks_failed_runs() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        service = ResearchAuditService(session)
        run_id = service.create_research_run(ticker="NVDA", question="What broke?")
        service.complete_research_run(
            run_id=run_id,
            ticker="NVDA",
            question="What broke?",
            latency_ms=12.5,
            error_message="boom",
        )

        row = session.scalar(select(ResearchRunORM).where(ResearchRunORM.run_id == run_id))
        assert row is not None
        assert row.status == "failed"
        assert row.error_message == "boom"

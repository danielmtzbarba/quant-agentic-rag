from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from stock_agent_rag.db import Base, ThesisArtifactORM
from stock_agent_rag.schemas import ContradictionRecord, EvidenceRecord
from stock_agent_rag.thesis_artifacts import ThesisArtifactService


class StubSettings:
    thesis_storage_provider = "local"
    thesis_artifact_bucket = "thesis-artifacts"
    thesis_artifact_local_mirror = True
    thesis_artifact_base_dir: Path
    embedding_model_name = "text-embedding-3-large"

    def __init__(self, base_dir: Path) -> None:
        self.thesis_artifact_base_dir = base_dir


def test_thesis_artifact_service_persists_markdown_and_metadata(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    settings = StubSettings(tmp_path)

    result = {
        "plan": "Review filings and sentiment.",
        "report": "NVDA thesis body [source:filing-1]",
        "verification_status": "pass",
        "verification_summary": "Verifier pass.",
        "verification_metrics": {
            "deterministic_status": "pass",
            "cited_retrieved_sources": 1,
            "citation_coverage": 1.0,
            "structured_findings": 2,
            "unsupported_findings": 0,
            "partially_grounded_findings": 0,
        },
        "model_metadata": {"models": ["gpt-4o-mini"]},
        "retrieved_evidence": [
            EvidenceRecord(
                source_id="filing-1",
                ticker="NVDA",
                title="10-Q",
                content="Demand remains strong.",
                document_type="filing",
                published_at=datetime(2026, 4, 18, tzinfo=UTC),
            )
        ],
        "retrieval_metrics": {"merged_retrieved_count": 1},
        "runtime_metrics": {"retry_count": 0, "timeout_count": 0},
        "estimated_cost_usd": 0.12,
        "latency_ms": 42.0,
        "contradictions": [
            ContradictionRecord(
                topic="demand_outlook",
                claim_a="Demand is strong",
                claim_b="Demand may slow",
                analyst_a="fundamentals",
                analyst_b="risk",
                evidence_ids_a=["filing-1"],
                evidence_ids_b=["news-1"],
            )
        ],
        "contradiction_summary": "Reviewed 1 material contradiction.",
    }

    with Session(engine, expire_on_commit=False) as session:
        summary = ThesisArtifactService(session, settings=settings).persist(
            run_id="run-123",
            ticker="NVDA",
            question="What is the thesis?",
            result=result,
        )

        row = session.scalar(select(ThesisArtifactORM))

    assert row is not None
    assert summary.run_id == "run-123"
    assert summary.bucket == "thesis-artifacts"
    assert summary.object_key.endswith("/run-123.md")
    assert summary.markdown_path is not None
    assert Path(summary.markdown_path).exists()
    assert row.ticker == "NVDA"
    assert row.verification_status == "pass"
    assert row.structured_findings_count == 2
    assert row.contradictions_count == 1
    assert row.top_source_ids_json == ["filing-1"]
    assert row.markdown_path == summary.markdown_path
    assert "NVDA Research Thesis" in Path(summary.markdown_path).read_text(encoding="utf-8")

from __future__ import annotations

from types import SimpleNamespace

import stock_agent_rag.workflow as workflow_module
from stock_agent_rag.schemas import (
    AnalystFinding,
    AnalystOutput,
    ContradictionRecord,
    EvidenceRecord,
)
from stock_agent_rag.workflow import contradiction_check_node, contradiction_review_node


def test_contradiction_check_detects_cross_analyst_conflict() -> None:
    state = {
        "fundamentals_analysis": AnalystOutput(
            summary="fundamentals",
            findings=[
                AnalystFinding(
                    finding="Gross margin improved on stronger data center demand",
                    evidence_ids=["filing-1"],
                    confidence=0.9,
                    missing_data=[],
                    finding_type="strength",
                )
            ],
        ),
        "risk_analysis": AnalystOutput(
            summary="risk",
            findings=[
                AnalystFinding(
                    finding="Gross margin faces pressure from weaker demand",
                    evidence_ids=["news-1"],
                    confidence=0.8,
                    missing_data=[],
                    finding_type="risk",
                )
            ],
        ),
    }

    result = contradiction_check_node(state)

    contradictions = result["contradictions"]
    assert len(contradictions) == 1
    assert contradictions[0].analyst_a == "fundamentals"
    assert contradictions[0].analyst_b == "risk"
    assert "demand" in contradictions[0].topic
    assert contradictions[0].resolution_status == "open"
    assert result["contradiction_summary"].startswith("Detected 1 cross-analyst contradictions")


def test_contradiction_check_ignores_same_polarity_findings() -> None:
    state = {
        "fundamentals_analysis": AnalystOutput(
            summary="fundamentals",
            findings=[
                AnalystFinding(
                    finding="Revenue growth remains strong in enterprise",
                    evidence_ids=["filing-1"],
                    confidence=0.8,
                    missing_data=[],
                    finding_type="strength",
                )
            ],
        ),
        "sentiment_analysis": AnalystOutput(
            summary="sentiment",
            findings=[
                AnalystFinding(
                    finding="Management tone on enterprise growth was constructive",
                    evidence_ids=["transcript-1"],
                    confidence=0.75,
                    missing_data=[],
                    finding_type="positive",
                )
            ],
        ),
    }

    result = contradiction_check_node(state)

    assert result["contradictions"] == []
    assert result["contradiction_summary"] == "No cross-analyst contradictions detected."


def test_contradiction_review_normalizes_shortlisted_pairs() -> None:
    class StubReviewModel:
        model_name = "gpt-test"
        temperature = 0.0

        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(
                    is_contradiction=True,
                    contradiction_kind="direct_conflict",
                    normalized_topic="gross_margin",
                    severity="high",
                    resolution_status="open",
                    rationale="Both claims address margin direction and conflict directly.",
                    supporting_evidence_ids_a=["filing-1"],
                    supporting_evidence_ids_b=["news-1"],
                ),
                "raw": SimpleNamespace(
                    usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
                    response_metadata={"model_name": "gpt-test"},
                ),
            }

    original = workflow_module._get_contradiction_review_model
    workflow_module._get_contradiction_review_model = lambda: StubReviewModel()

    state = {
        "contradictions": [
            ContradictionRecord(
                topic="demand / margin",
                claim_a="Gross margin improved on stronger data center demand",
                claim_b="Gross margin faces pressure from weaker demand",
                analyst_a="fundamentals",
                analyst_b="risk",
                evidence_ids_a=["filing-1"],
                evidence_ids_b=["news-1"],
            )
        ],
        "retrieved_evidence": [
            EvidenceRecord(
                source_id="filing-1",
                ticker="NVDA",
                title="10-Q",
                content="Gross margin improved due to data center mix.",
                document_type="filing",
            ),
            EvidenceRecord(
                source_id="news-1",
                ticker="NVDA",
                title="Reuters",
                content="Gross margin may face pressure if demand softens.",
                document_type="news",
            ),
        ],
    }

    try:
        result = contradiction_review_node(state)
    finally:
        workflow_module._get_contradiction_review_model = original

    assert len(result["contradictions"]) == 1
    reviewed = result["contradictions"][0]
    assert reviewed.topic == "gross_margin"
    assert reviewed.contradiction_kind == "direct_conflict"
    assert reviewed.rationale
    assert result["contradiction_summary"].startswith("Reviewed 1 material contradictions")
    assert result["node_metrics"]["contradiction_review"]["total_tokens"] == 30


def test_contradiction_review_falls_back_to_deterministic_normalization() -> None:
    original = workflow_module._get_contradiction_review_model
    workflow_module._get_contradiction_review_model = lambda: (_ for _ in ()).throw(RuntimeError())

    state = {
        "contradictions": [
            ContradictionRecord(
                topic="demand / next quarter",
                claim_a="Demand improved this quarter with strong orders",
                claim_b="Demand may weaken next quarter as orders normalize",
                analyst_a="fundamentals",
                analyst_b="risk",
                evidence_ids_a=["filing-1"],
                evidence_ids_b=["news-1"],
            )
        ]
    }

    try:
        result = contradiction_review_node(state)
    finally:
        workflow_module._get_contradiction_review_model = original

    assert len(result["contradictions"]) == 1
    reviewed = result["contradictions"][0]
    assert reviewed.topic == "demand_outlook"
    assert reviewed.contradiction_kind == "time_horizon_tension"
    assert reviewed.resolution_status == "explained"

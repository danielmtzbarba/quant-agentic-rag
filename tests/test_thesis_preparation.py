from __future__ import annotations

from datetime import UTC, datetime

from stock_agent_rag.schemas import (
    AnalystFinding,
    AnalystOutput,
    EvidenceRecord,
    FundamentalsSnapshot,
)
from stock_agent_rag.tools import fundamentals_snapshot_to_evidence
from stock_agent_rag.workflow import (
    _render_thesis_grounding_packet,
    aggregate_evidence_node,
    thesis_preparation_node,
)


def test_thesis_preparation_buckets_findings_into_sections() -> None:
    state = {
        "fundamentals_analysis": AnalystOutput(
            summary="fundamentals summary",
            findings=[
                AnalystFinding(
                    finding="Revenue growth remains strong",
                    evidence_ids=["filing-1"],
                    confidence=0.9,
                    missing_data=[],
                    finding_type="strength",
                ),
                AnalystFinding(
                    finding="Margin pressure may compress earnings",
                    evidence_ids=["filing-2"],
                    confidence=0.7,
                    missing_data=["segment margin breakout"],
                    finding_type="weakness",
                ),
            ],
            evidence_gaps=["Need updated segment disclosure"],
            overall_confidence=0.8,
        ),
        "sentiment_analysis": AnalystOutput(
            summary="sentiment summary",
            findings=[
                AnalystFinding(
                    finding="Management tone is constructive",
                    evidence_ids=["transcript-1"],
                    confidence=0.8,
                    missing_data=[],
                    finding_type="positive",
                )
            ],
            overall_confidence=0.75,
        ),
        "risk_analysis": AnalystOutput(
            summary="risk summary",
            findings=[
                AnalystFinding(
                    finding="Export controls remain a material risk",
                    evidence_ids=["filing-3", "news-1"],
                    confidence=0.85,
                    missing_data=[],
                    finding_type="risk",
                )
            ],
            overall_confidence=0.85,
        ),
    }

    result = thesis_preparation_node(state)
    preparation = result["thesis_preparation"]
    sections = {section.section_id: section for section in preparation.sections}

    assert "executive_summary" in sections
    assert "bull_case" in sections
    assert "bear_case" in sections
    assert "key_risks" in sections
    assert "evidence_gaps" in sections

    assert any(
        f.finding == "Revenue growth remains strong" for f in sections["bull_case"].findings
    )
    assert any(
        f.finding == "Management tone is constructive" for f in sections["bull_case"].findings
    )
    assert any(
        f.finding == "Margin pressure may compress earnings"
        for f in sections["bear_case"].findings
    )
    assert any(
        f.finding == "Export controls remain a material risk"
        for f in sections["key_risks"].findings
    )
    assert any(
        "Need updated segment disclosure" in f.finding for f in sections["evidence_gaps"].findings
    )
    assert "filing-1" in sections["executive_summary"].evidence_ids


def test_fundamentals_snapshot_to_evidence_creates_stable_metric_records() -> None:
    snapshot = FundamentalsSnapshot(
        ticker="NVDA",
        as_of=datetime(2026, 4, 18, tzinfo=UTC),
        metrics={
            "revenue_growth": 0.732,
            "debt_to_equity": 7.255,
            "current_ratio": None,
        },
        source="yfinance",
    )

    records = fundamentals_snapshot_to_evidence(snapshot)

    assert [record.source_id for record in records] == [
        "nvda-fundamentals-revenue_growth",
        "nvda-fundamentals-debt_to_equity",
    ]
    assert all(record.document_type == "fundamentals" for record in records)
    assert all(record.provider == "yfinance" for record in records)
    assert "Metric: revenue_growth" in records[0].content


def test_aggregate_evidence_includes_fundamentals_snapshot_records() -> None:
    state = {
        "fundamentals": FundamentalsSnapshot(
            ticker="NVDA",
            as_of=datetime(2026, 4, 18, tzinfo=UTC),
            metrics={"revenue_growth": 0.732},
            source="yfinance",
        ),
        "fundamentals_evidence": [],
        "sentiment_evidence": [],
        "risk_evidence": [
            EvidenceRecord(
                source_id="risk-1",
                ticker="NVDA",
                title="Risk",
                content="Export controls remain a risk.",
                document_type="filing",
            )
        ],
    }

    result = aggregate_evidence_node(state)

    assert [record.source_id for record in result["retrieved_evidence"]] == [
        "nvda-fundamentals-revenue_growth",
        "risk-1",
    ]


def test_render_thesis_grounding_packet_uses_section_structure_and_snippets() -> None:
    state = {
        "fundamentals_analysis": AnalystOutput(
            summary="fundamentals summary",
            findings=[
                AnalystFinding(
                    finding="Revenue growth remains strong",
                    evidence_ids=["filing-1"],
                    confidence=0.9,
                    missing_data=[],
                    finding_type="strength",
                )
            ],
            overall_confidence=0.8,
        ),
        "risk_analysis": AnalystOutput(
            summary="risk summary",
            findings=[
                AnalystFinding(
                    finding="Export controls remain a material risk",
                    evidence_ids=["risk-1"],
                    confidence=0.85,
                    missing_data=["customer concentration update"],
                    finding_type="risk",
                )
            ],
            overall_confidence=0.85,
        ),
        "retrieved_evidence": [
            EvidenceRecord(
                source_id="filing-1",
                ticker="NVDA",
                title="10-Q MDA",
                content="Revenue growth accelerated due to data center demand.",
                document_type="filing",
            ),
            EvidenceRecord(
                source_id="risk-1",
                ticker="NVDA",
                title="Risk Factors",
                content="Export controls remain a material headwind for China sales.",
                document_type="filing",
            ),
        ],
    }
    state.update(thesis_preparation_node(state))

    packet = _render_thesis_grounding_packet(state)

    assert "Section ID: executive_summary" in packet
    assert "Allowed section evidence ids: [source:filing-1], [source:risk-1]" in packet
    assert "1. finding=Revenue growth remains strong" in packet
    assert "allowed_evidence_ids=[source:filing-1]" in packet
    assert "snippet [source:filing-1] filing | 10-Q MDA | Revenue growth accelerated" in packet
    assert "missing_data=customer concentration update" in packet

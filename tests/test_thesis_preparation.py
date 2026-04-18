from __future__ import annotations

from stock_agent_rag.schemas import AnalystFinding, AnalystOutput
from stock_agent_rag.workflow import thesis_preparation_node


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

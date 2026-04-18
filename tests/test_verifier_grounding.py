from __future__ import annotations

from stock_agent_rag.schemas import AnalystFinding, AnalystOutput
from stock_agent_rag.workflow import (
    _extract_cited_source_ids,
    _structured_grounding_metrics,
    _structured_grounding_summary,
)


def test_extract_cited_source_ids_parses_inline_citations() -> None:
    report = "Bull case [source:a1] and risk note [source:b2]. Repeat [source:a1]."

    cited = _extract_cited_source_ids(report)

    assert cited == {"a1", "b2"}


def test_structured_grounding_summary_counts_grounded_and_unsupported_findings() -> None:
    state = {
        "fundamentals_analysis": AnalystOutput(
            summary="summary",
            findings=[
                AnalystFinding(
                    finding="Revenue growth improved",
                    evidence_ids=["filing-1"],
                    confidence=0.8,
                    missing_data=[],
                ),
                AnalystFinding(
                    finding="Margin durability is unclear",
                    evidence_ids=["filing-2", "news-1"],
                    confidence=0.6,
                    missing_data=["segment margin breakout"],
                ),
            ],
        ),
        "sentiment_analysis": AnalystOutput(
            summary="summary",
            findings=[
                AnalystFinding(
                    finding="Management tone is constructive",
                    evidence_ids=[],
                    confidence=0.7,
                    missing_data=["more transcript coverage"],
                )
            ],
        ),
    }
    report = "Growth is strong [source:filing-1]. Risks remain [source:filing-2]."

    metrics = _structured_grounding_metrics(state, report)
    summary = _structured_grounding_summary(metrics)

    assert "grounded_findings=1" in summary
    assert "partially_grounded_findings=1" in summary
    assert "unsupported_findings=1" in summary
    assert "findings_without_evidence_ids=1" in summary

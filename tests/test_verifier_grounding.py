from __future__ import annotations

from stock_agent_rag.schemas import AnalystFinding, AnalystOutput
from stock_agent_rag.workflow import (
    _extract_cited_source_ids,
    _structured_grounding_metrics,
    _structured_grounding_summary,
    validate_thesis_report,
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


def test_validate_thesis_report_flags_uncited_numeric_claims() -> None:
    report = "Revenue grew 73.2% year over year without a citation."

    errors = validate_thesis_report(report)

    assert any("uncited numeric claims" in error for error in errors)


def test_validate_thesis_report_flags_prohibited_placeholder_text() -> None:
    report = "Debt to equity is elevated (evidence not provided)."

    errors = validate_thesis_report(report)

    assert any("evidence not provided" in error for error in errors)


def test_validate_thesis_report_flags_malformed_source_citations() -> None:
    report = "Revenue grew 73.2% year over year (source:filing-1)."

    errors = validate_thesis_report(report)

    assert any("malformed citations" in error for error in errors)


def test_validate_thesis_report_allows_cited_numeric_claims() -> None:
    report = "Revenue grew 73.2% year over year [source:filing-1]."

    errors = validate_thesis_report(report)

    assert errors == []

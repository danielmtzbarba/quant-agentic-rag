from __future__ import annotations

from pathlib import Path

import pytest

from stock_agent_rag.config import Settings
from stock_agent_rag.ingestion.sec import SecFilingIngestionService

SAMPLE_FILING = """
<html>
  <body>
    <div>Filed as of Date: 20240131</div>
    <div>Central Index Key: 0000320193</div>
    <h1>Item 1. Business</h1>
    <p>Apple designs, manufactures, and markets smartphones, personal computers, tablets,
    wearables, and accessories. The company also sells services and digital content globally.</p>
    <p>The business section provides background, market focus,
    and product strategy details for investors.</p>
    <h1>Item 1A. Risk Factors</h1>
    <p>Adverse global macroeconomic conditions can materially affect demand and operations.
    Supply chain disruption, competition, regulatory changes,
    and litigation remain meaningful risks.</p>
    <p>Management also notes dependence on third-party manufacturing
    and component availability.</p>
    <h1>Item 7. Management's Discussion and Analysis
    of Financial Condition and Results of Operations</h1>
    <p>Net sales increased year over year driven by iPhone and Services performance.
    Gross margin expansion reflected product mix and operating leverage across major segments.</p>
    <p>Cash flow remained strong and management discussed capital return,
    liquidity, and working capital.</p>
    <h1>Item 8. Financial Statements and Supplementary Data</h1>
    <p>Consolidated statements of operations, balance sheets,
    and cash flows follow with notes and schedules.</p>
  </body>
</html>
"""


def test_extract_10k_sections() -> None:
    service = SecFilingIngestionService(settings=Settings())
    cleaned = service._clean_filing_text(SAMPLE_FILING)

    sections = service.extract_sections(cleaned, form_type="10-K")
    section_ids = [section.section_id for section in sections]

    assert "item_1_business" in section_ids
    assert "item_1a_risk_factors" in section_ids
    assert "item_7_mda" in section_ids
    assert "item_8_financial_statements" in section_ids


def test_chunk_document_generates_filing_chunks(tmp_path: Path) -> None:
    service = SecFilingIngestionService(settings=Settings())
    path = tmp_path / "full-submission.txt"
    path.write_text(SAMPLE_FILING, encoding="utf-8")

    document = service._build_document_record(path=path, ticker="AAPL", form_type="10-K")
    chunks = service._chunk_document(document)

    assert chunks
    assert all(chunk.document_type == "filing" for chunk in chunks)
    assert any(chunk.section == "item_7_mda" for chunk in chunks)
    assert all(chunk.metadata_version == service.settings.sec_metadata_version for chunk in chunks)


def test_validate_sec_identity_rejects_placeholder_email() -> None:
    settings = Settings(
        SEC_COMPANY_NAME="Quant Agentic RAG",
        SEC_EMAIL_ADDRESS="you@example.com",
    )
    service = SecFilingIngestionService(settings=settings)

    with pytest.raises(ValueError):
        service._validate_sec_identity()

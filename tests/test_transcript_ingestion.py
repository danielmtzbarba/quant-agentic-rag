from __future__ import annotations

import json
from pathlib import Path

from stock_agent_rag.config import Settings
from stock_agent_rag.ingestion.transcripts import (
    AlphaVantageTranscriptIngestionService,
    TranscriptFetchResult,
)

SAMPLE_TRANSCRIPT_PAYLOAD = {
    "symbol": "NVDA",
    "quarter": "2025Q4",
    "date": "2025-02-26 00:00:00",
    "transcript": """
    John Doe -- Chief Executive Officer
    Jane Smith -- Chief Financial Officer

    John Doe:
    Demand remained strong across data center and gaming.
    We continue to see robust enterprise adoption.

    Jane Smith:
    Gross margins improved year over year and operating cash flow remained healthy.

    Alex Analyst:
    Can you talk about supply constraints heading into next quarter?

    John Doe:
    Supply is improving, but we still expect some tightness in advanced packaging.
    """,
}

SAMPLE_TRANSCRIPT_LIST_PAYLOAD = {
    "symbol": "NVDA",
    "quarter": "2026Q1",
    "date": "2026-05-28 00:00:00",
    "transcript": [
        {
            "speaker": "Operator",
            "content": "Good afternoon and welcome to the conference call.",
            "sentiment": "0.0",
        },
        {
            "speaker": "Jensen Huang",
            "content": "Demand remained strong across AI infrastructure and enterprise workloads.",
            "sentiment": "0.7",
        },
        {
            "speaker": "Colette Kress",
            "content": "Gross margins expanded and cash generation remained healthy.",
            "sentiment": "0.5",
        },
    ],
}


def test_extract_transcript_turns_preserves_speakers() -> None:
    service = AlphaVantageTranscriptIngestionService(settings=Settings(VANTAGE_API_KEY="test-key"))

    turns = service.extract_transcript_turns(SAMPLE_TRANSCRIPT_PAYLOAD["transcript"])

    assert len(turns) == 4
    assert turns[0].speaker == "John Doe"
    assert turns[0].speaker_role == "Chief Executive Officer"
    assert turns[1].speaker == "Jane Smith"
    assert turns[2].speaker == "Alex Analyst"


def test_ingest_transcript_writes_normalized_and_chunk_files(tmp_path: Path) -> None:
    settings = Settings(
        DATA_DIR=str(tmp_path),
        RAG_CORPUS_DIR=str(tmp_path / "chunks"),
        DATABASE_URL="",
        VANTAGE_API_KEY="test-key",
        TRANSCRIPT_METADATA_VERSION="1.0",
    )
    service = AlphaVantageTranscriptIngestionService(settings=settings)

    service._fetch_transcript_payload = (  # type: ignore[method-assign]
        lambda **_: TranscriptFetchResult(
            payload=SAMPLE_TRANSCRIPT_PAYLOAD.copy(),
            request_url="https://example.invalid/fake",
            http_status=200,
            fetched_at=service._parse_datetime("2025-02-26 00:00:00"),
        )
    )

    summary = service.ingest(ticker="NVDA", year=2025, quarter=4)

    assert summary.processed_documents == 1
    assert summary.chunk_count >= 4

    normalized_path = Path(summary.normalized_paths[0])
    chunk_path = Path(summary.chunk_paths[0])
    assert normalized_path.exists()
    assert chunk_path.exists()

    normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
    assert normalized_payload["source_type"] == "transcript"
    assert normalized_payload["transcript_turns"][0]["speaker"] == "John Doe"

    chunk_lines = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines()]
    assert chunk_lines[0]["document_type"] == "transcript"
    assert chunk_lines[0]["speaker"] == "John Doe"


def test_ingest_transcript_uses_cache_when_raw_exists(tmp_path: Path) -> None:
    settings = Settings(
        DATA_DIR=str(tmp_path),
        RAG_CORPUS_DIR=str(tmp_path / "chunks"),
        DATABASE_URL="",
        VANTAGE_API_KEY="test-key",
        TRANSCRIPT_METADATA_VERSION="1.0",
    )
    service = AlphaVantageTranscriptIngestionService(settings=settings)

    service._persist_raw_payload(  # type: ignore[attr-defined]
        ticker="NVDA",
        year=2025,
        quarter=4,
        payload=SAMPLE_TRANSCRIPT_PAYLOAD.copy(),
        request_url="https://example.invalid/fake",
        http_status=200,
        fetched_at=service._parse_datetime("2025-02-26 00:00:00"),
    )

    def _should_not_fetch(**_kwargs):
        raise AssertionError("Expected cached payload usage; network fetch should not run.")

    service._fetch_transcript_payload = _should_not_fetch  # type: ignore[method-assign]

    summary = service.ingest(ticker="NVDA", year=2025, quarter=4)
    assert summary.processed_documents == 1
    assert summary.chunk_count >= 4


def test_validate_vantage_payload_requires_transcript_content() -> None:
    service = AlphaVantageTranscriptIngestionService(settings=Settings(VANTAGE_API_KEY="test-key"))

    try:
        service._validate_transcript_payload({"quarter": "2025Q4", "transcript": "   "})
        raise AssertionError("Expected validation to fail")
    except RuntimeError as exc:
        assert "validation failed" in str(exc).lower()


def test_validate_vantage_payload_accepts_structured_transcript_list() -> None:
    service = AlphaVantageTranscriptIngestionService(settings=Settings(VANTAGE_API_KEY="test-key"))

    payload = service._validate_transcript_payload(SAMPLE_TRANSCRIPT_LIST_PAYLOAD)

    assert isinstance(payload.transcript, str)
    assert "Jensen Huang" in payload.transcript


def test_build_document_record_uses_structured_transcript_turns() -> None:
    service = AlphaVantageTranscriptIngestionService(settings=Settings(VANTAGE_API_KEY="test-key"))

    document = service._build_document_record(
        raw_payload=SAMPLE_TRANSCRIPT_LIST_PAYLOAD,
        raw_path="/tmp/fake.json",
        ticker="NVDA",
        year=2026,
        quarter=1,
    )

    assert len(document.transcript_turns) == 3
    assert document.transcript_turns[1].speaker == "Jensen Huang"
    assert "AI infrastructure" in document.cleaned_text

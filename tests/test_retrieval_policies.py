from __future__ import annotations

import json
from pathlib import Path

from stock_agent_rag.tools import local_corpus_search, merge_evidence_sets


def test_profiled_retrieval_filters_by_expected_source_types(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "chunks"
    (corpus_dir / "news" / "NVDA").mkdir(parents=True, exist_ok=True)
    (corpus_dir / "transcripts" / "NVDA").mkdir(parents=True, exist_ok=True)
    (corpus_dir / "sec" / "NVDA" / "10-K").mkdir(parents=True, exist_ok=True)

    filing_path = corpus_dir / "sec" / "NVDA" / "10-K" / "nvda-filing.jsonl"
    filing_path.write_text(
        json.dumps(
            {
                "source_id": "filing-1",
                "ticker": "NVDA",
                "title": "Risk Factors",
                "content": "Item 1A. Risk factors include supply concentration and regulation.",
                "document_type": "filing",
                "section": "item_1a_risk_factors",
            }
        ),
        encoding="utf-8",
    )

    transcript_path = corpus_dir / "transcripts" / "NVDA" / "nvda-transcript.jsonl"
    transcript_path.write_text(
        json.dumps(
            {
                "source_id": "transcript-1",
                "ticker": "NVDA",
                "title": "Q1 Transcript",
                "content": "Management discussed enterprise demand and guidance.",
                "document_type": "transcript",
                "speaker": "Jensen Huang",
                "speaker_role": "Chief Executive Officer",
            }
        ),
        encoding="utf-8",
    )

    news_path = corpus_dir / "news" / "NVDA" / "nvda-news.jsonl"
    news_path.write_text(
        json.dumps(
            {
                "source_id": "news-1",
                "ticker": "NVDA",
                "title": "NVIDIA faces export questions",
                "content": "News coverage highlighted regulatory pressure and sentiment shifts.",
                "document_type": "news",
                "publisher": "Reuters",
                "sentiment_label": "Bearish",
                "sentiment_score": -0.4,
            }
        ),
        encoding="utf-8",
    )

    original = local_corpus_search.__globals__["get_settings"]
    local_corpus_search.__globals__["get_settings"] = lambda: type(
        "StubSettings",
        (),
        {"corpus_dir": corpus_dir, "default_top_k": 4},
    )()

    try:
        fundamentals = local_corpus_search(
            query="NVDA financial statements mda",
            ticker="NVDA",
            profile="fundamentals",
        )
        sentiment = local_corpus_search(
            query="NVDA management guidance sentiment",
            ticker="NVDA",
            profile="sentiment",
        )
        risk = local_corpus_search(
            query="NVDA risk regulation export pressure",
            ticker="NVDA",
            profile="risk",
        )
    finally:
        local_corpus_search.__globals__["get_settings"] = original

    assert all(record.document_type == "filing" for record in fundamentals)
    assert all(record.document_type in {"transcript", "news"} for record in sentiment)
    assert any(record.document_type == "filing" for record in risk)
    assert any(record.document_type == "news" for record in risk)


def test_merge_evidence_sets_deduplicates_and_keeps_highest_score() -> None:
    from stock_agent_rag.schemas import EvidenceRecord

    merged = merge_evidence_sets(
        [
            EvidenceRecord(
                source_id="same",
                ticker="NVDA",
                title="Doc A",
                content="foo",
                document_type="filing",
                score=1.0,
            )
        ],
        [
            EvidenceRecord(
                source_id="same",
                ticker="NVDA",
                title="Doc A",
                content="foo",
                document_type="filing",
                score=3.0,
            ),
            EvidenceRecord(
                source_id="other",
                ticker="NVDA",
                title="Doc B",
                content="bar",
                document_type="news",
                score=2.0,
            ),
        ],
    )

    assert [record.source_id for record in merged] == ["same", "other"]
    assert merged[0].score == 3.0

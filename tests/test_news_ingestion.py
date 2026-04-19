from __future__ import annotations

import json
from pathlib import Path

from stock_agent_rag.config import Settings
from stock_agent_rag.ingestion.news import AlphaVantageNewsIngestionService, NewsFetchResult

SAMPLE_NEWS_PAYLOAD = {
    "feed": [
        {
            "title": "NVIDIA launches new AI systems for enterprise customers",
            "url": "https://example.com/nvda-ai-systems",
            "time_published": "20260416T120000",
            "summary": (
                "The company unveiled new enterprise offerings and highlighted "
                "strong demand."
            ),
            "source": "Reuters",
            "overall_sentiment_score": "0.42",
            "overall_sentiment_label": "Bullish",
            "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.9"}],
        },
        {
            "title": "NVIDIA supply chain remains a focus for investors",
            "url": "https://example.com/nvda-supply-chain",
            "time_published": "20260416T130000",
            "summary": "Analysts continue monitoring advanced packaging constraints.",
            "source": "Bloomberg",
            "overall_sentiment_score": "0.05",
            "overall_sentiment_label": "Neutral",
            "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.8"}],
        },
        {
            "title": "Intel expands AI laptop lineup",
            "url": "https://example.com/intel-ai-laptops",
            "time_published": "20260416T140000",
            "summary": "Intel highlighted stronger PC demand and battery performance.",
            "source": "Reuters",
            "overall_sentiment_score": "0.15",
            "overall_sentiment_label": "Somewhat-Bullish",
            "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.9"}],
        },
    ]
}


def test_extract_articles_filters_and_validates() -> None:
    service = AlphaVantageNewsIngestionService(settings=Settings(VANTAGE_API_KEY="test-key"))

    articles = service._extract_articles(SAMPLE_NEWS_PAYLOAD, ticker="NVDA")

    assert len(articles) == 2
    assert articles[0].source == "Reuters"
    assert all("Intel expands" not in article.title for article in articles)


def test_ingest_news_writes_normalized_and_chunk_files(tmp_path: Path) -> None:
    settings = Settings(
        DATA_DIR=str(tmp_path),
        RAG_CORPUS_DIR=str(tmp_path / "chunks"),
        DATABASE_URL="",
        VANTAGE_API_KEY="test-key",
        NEWS_METADATA_VERSION="1.0",
    )
    service = AlphaVantageNewsIngestionService(settings=settings)

    service._fetch_news_payload = (  # type: ignore[method-assign]
        lambda **_: NewsFetchResult(
            payload=SAMPLE_NEWS_PAYLOAD.copy(),
            request_url="https://example.invalid/news",
            http_status=200,
            fetched_at=service._parse_datetime("2026-04-16 12:00:00"),
        )
    )

    summary = service.ingest(ticker="NVDA", limit=2)

    assert summary.processed_documents == 2
    assert summary.chunk_count == 2

    normalized_path = Path(summary.normalized_paths[0])
    chunk_path = Path(summary.chunk_paths[0])
    assert normalized_path.exists()
    assert chunk_path.exists()

    normalized_payload = json.loads(normalized_path.read_text(encoding="utf-8"))
    assert normalized_payload["source_type"] == "news"
    assert normalized_payload["publisher"] == "Reuters"
    assert normalized_payload["sentiment_label"] == "Bullish"
    assert normalized_payload["entity_title_match"] is True
    assert normalized_payload["news_relevance_tier"] == "direct"
    assert normalized_payload["source_quality_tier"] == "trusted"

    chunk_lines = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines()]
    assert chunk_lines[0]["document_type"] == "news"
    assert chunk_lines[0]["publisher"] == "Reuters"
    assert chunk_lines[0]["sentiment_score"] == 0.42
    assert chunk_lines[0]["news_relevance_score"] >= 0.7
    assert chunk_lines[0]["source_quality_tier"] == "trusted"


def test_ingest_news_uses_cache_when_raw_exists(tmp_path: Path) -> None:
    settings = Settings(
        DATA_DIR=str(tmp_path),
        RAG_CORPUS_DIR=str(tmp_path / "chunks"),
        DATABASE_URL="",
        VANTAGE_API_KEY="test-key",
        NEWS_METADATA_VERSION="1.0",
    )
    service = AlphaVantageNewsIngestionService(settings=settings)

    service._persist_raw_payload(
        ticker="NVDA",
        limit=2,
        payload=SAMPLE_NEWS_PAYLOAD.copy(),
        request_url="https://example.invalid/news",
        http_status=200,
        fetched_at=service._parse_datetime("2026-04-16 12:00:00"),
    )

    def _should_not_fetch(**_kwargs):
        raise AssertionError("Expected cached payload usage; network fetch should not run.")

    service._fetch_news_payload = _should_not_fetch  # type: ignore[method-assign]

    summary = service.ingest(ticker="NVDA", limit=2)
    assert summary.processed_documents == 2
    assert summary.chunk_count == 2


def test_assess_article_relevance_requires_direct_entity_mention() -> None:
    service = AlphaVantageNewsIngestionService(settings=Settings(VANTAGE_API_KEY="test-key"))

    direct_article = service._extract_articles(
        {
            "feed": [
                {
                    "title": "NVIDIA wins new hyperscaler deployment",
                    "url": "https://example.com/nvda-direct",
                    "time_published": "20260416T120000",
                    "summary": "NVIDIA said enterprise demand remains strong.",
                    "source": "Reuters",
                    "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.95"}],
                }
            ]
        },
        ticker="NVDA",
    )[0]
    assessment = service._assess_article_relevance(article=direct_article, ticker="NVDA")

    assert assessment is not None
    assert assessment.entity_title_match is True
    assert assessment.news_relevance_tier == "direct"
    assert assessment.source_quality_tier == "trusted"

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stock_agent_rag.schemas import EvidenceRecord
from stock_agent_rag.telemetry import (
    aggregate_estimated_cost_usd,
    aggregate_runtime_metrics,
    build_retrieval_metrics,
    estimate_cost_usd,
)


def test_estimate_cost_usd_uses_known_model_pricing() -> None:
    estimated = estimate_cost_usd(
        model_name="gpt-4o-mini",
        input_tokens=1000,
        output_tokens=500,
    )

    assert estimated == 0.00045


def test_aggregate_runtime_metrics_and_cost() -> None:
    node_metrics = {
        "planner": {
            "retry_count": 1,
            "timeout_count": 0,
            "estimated_cost_usd": 0.00012,
        },
        "thesis": {
            "retry_count": 0,
            "timeout_count": 1,
            "estimated_cost_usd": 0.00034,
        },
    }

    assert aggregate_runtime_metrics(node_metrics) == {
        "retry_count": 1,
        "timeout_count": 1,
    }
    assert aggregate_estimated_cost_usd(node_metrics) == 0.00046


def test_build_retrieval_metrics_includes_hit_rates_and_freshness() -> None:
    now = datetime.now(UTC)
    filings = [
        EvidenceRecord(
            source_id="filing-1",
            ticker="NVDA",
            title="10-K",
            content="content",
            document_type="filing",
            published_at=now - timedelta(days=2),
        )
    ]
    sentiment = [
        EvidenceRecord(
            source_id="news-1",
            ticker="NVDA",
            title="News",
            content="content",
            document_type="news",
            published_at=now - timedelta(hours=12),
        )
    ]
    risk = [
        EvidenceRecord(
            source_id="transcript-1",
            ticker="NVDA",
            title="Transcript",
            content="content",
            document_type="transcript",
            published_at=now - timedelta(days=40),
        )
    ]
    merged = filings + sentiment + risk

    metrics = build_retrieval_metrics(
        fundamentals_evidence=filings,
        sentiment_evidence=sentiment,
        risk_evidence=risk,
        retrieved_evidence=merged,
        default_top_k=4,
        target_ticker="NVDA",
    )

    assert metrics["profile_hit_rates"] == {
        "fundamentals": 0.25,
        "sentiment": 0.25,
        "risk": 0.25,
    }
    assert metrics["source_type_counts"] == {
        "filing": 1,
        "news": 1,
        "transcript": 1,
    }
    assert metrics["sources_with_timestamps"] == 3
    assert metrics["fresh_sources_7d_ratio"] == 0.6667
    assert metrics["fresh_sources_30d_ratio"] == 0.6667
    assert metrics["off_ticker_evidence_count"] == 0
    assert metrics["off_ticker_evidence_rate"] == 0.0


def test_build_retrieval_metrics_tracks_off_ticker_news() -> None:
    news = [
        EvidenceRecord(
            source_id="news-1",
            ticker="NVDA",
            title="Direct company article",
            content="content",
            document_type="news",
            entity_title_match=True,
            entity_body_match=True,
        ),
        EvidenceRecord(
            source_id="news-2",
            ticker="INTC",
            title="Off ticker article",
            content="content",
            document_type="news",
            entity_title_match=False,
            entity_body_match=False,
        ),
    ]

    metrics = build_retrieval_metrics(
        fundamentals_evidence=[],
        sentiment_evidence=news,
        risk_evidence=[],
        retrieved_evidence=news,
        default_top_k=4,
        target_ticker="NVDA",
    )

    assert metrics["off_ticker_evidence_count"] == 1
    assert metrics["off_ticker_evidence_rate"] == 0.5

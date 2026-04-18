from __future__ import annotations

from datetime import UTC, datetime
from statistics import median
from typing import Any

from .schemas import EvidenceRecord

MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


def estimate_cost_usd(
    *,
    model_name: str | None,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    if not model_name:
        return None

    pricing = _resolve_model_pricing(model_name)
    if pricing is None:
        return None

    estimated = (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )
    return round(estimated, 8)


def aggregate_token_usage(node_metrics: dict[str, dict[str, object]]) -> dict[str, int]:
    return {
        "input_tokens": sum(
            int(metrics.get("input_tokens", 0)) for metrics in node_metrics.values()
        ),
        "output_tokens": sum(
            int(metrics.get("output_tokens", 0)) for metrics in node_metrics.values()
        ),
        "total_tokens": sum(
            int(metrics.get("total_tokens", 0)) for metrics in node_metrics.values()
        ),
    }


def collect_model_metadata(node_metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    models = sorted(
        {
            str(metrics.get("model_name"))
            for metrics in node_metrics.values()
            if metrics.get("model_name")
        }
    )
    providers = sorted(
        {
            str(metrics.get("provider"))
            for metrics in node_metrics.values()
            if metrics.get("provider")
        }
    )
    temperatures = sorted(
        {
            float(metrics.get("temperature"))
            for metrics in node_metrics.values()
            if metrics.get("temperature") is not None
        }
    )
    return {
        "models": models,
        "providers": providers,
        "temperatures": temperatures,
    }


def aggregate_runtime_metrics(node_metrics: dict[str, dict[str, object]]) -> dict[str, int]:
    return {
        "retry_count": sum(int(metrics.get("retry_count", 0)) for metrics in node_metrics.values()),
        "timeout_count": sum(
            int(metrics.get("timeout_count", 0)) for metrics in node_metrics.values()
        ),
    }


def aggregate_estimated_cost_usd(node_metrics: dict[str, dict[str, object]]) -> float | None:
    values = [
        float(metrics.get("estimated_cost_usd"))
        for metrics in node_metrics.values()
        if metrics.get("estimated_cost_usd") is not None
    ]
    if not values:
        return None
    return round(sum(values), 8)


def build_retrieval_metrics(
    *,
    fundamentals_evidence: list[EvidenceRecord],
    sentiment_evidence: list[EvidenceRecord],
    risk_evidence: list[EvidenceRecord],
    retrieved_evidence: list[EvidenceRecord],
    default_top_k: int,
) -> dict[str, object]:
    source_type_counts: dict[str, int] = {}
    document_ids: set[str] = set()
    publishers: set[str] = set()
    for record in retrieved_evidence:
        source_type_counts[record.document_type] = (
            source_type_counts.get(record.document_type, 0) + 1
        )
        if record.document_id:
            document_ids.add(record.document_id)
        if record.publisher:
            publishers.add(record.publisher)

    freshness_metrics = _build_freshness_metrics(retrieved_evidence)
    return {
        "profile_retrieved_counts": {
            "fundamentals": len(fundamentals_evidence),
            "sentiment": len(sentiment_evidence),
            "risk": len(risk_evidence),
        },
        "profile_hit_rates": {
            "fundamentals": round(_safe_ratio(len(fundamentals_evidence), default_top_k), 4),
            "sentiment": round(_safe_ratio(len(sentiment_evidence), default_top_k), 4),
            "risk": round(_safe_ratio(len(risk_evidence), default_top_k), 4),
        },
        "merged_retrieved_count": len(retrieved_evidence),
        "merged_hit_rate": round(
            _safe_ratio(len(retrieved_evidence), default_top_k * 3),
            4,
        ),
        "source_type_counts": source_type_counts,
        "unique_document_count": len(document_ids),
        "unique_publisher_count": len(publishers),
        **freshness_metrics,
    }


def _build_freshness_metrics(records: list[EvidenceRecord]) -> dict[str, object]:
    now = datetime.now(UTC)
    age_hours: list[float] = []
    for record in records:
        if record.published_at is None:
            continue
        published_at = record.published_at
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=UTC)
        age_hours.append(max((now - published_at).total_seconds() / 3600, 0.0))

    if not age_hours:
        return {
            "sources_with_timestamps": 0,
            "newest_source_age_hours": None,
            "oldest_source_age_hours": None,
            "median_source_age_hours": None,
            "fresh_sources_7d_ratio": None,
            "fresh_sources_30d_ratio": None,
        }

    fresh_7d = sum(age <= 24 * 7 for age in age_hours)
    fresh_30d = sum(age <= 24 * 30 for age in age_hours)
    return {
        "sources_with_timestamps": len(age_hours),
        "newest_source_age_hours": round(min(age_hours), 2),
        "oldest_source_age_hours": round(max(age_hours), 2),
        "median_source_age_hours": round(float(median(age_hours)), 2),
        "fresh_sources_7d_ratio": round(_safe_ratio(fresh_7d, len(age_hours)), 4),
        "fresh_sources_30d_ratio": round(_safe_ratio(fresh_30d, len(age_hours)), 4),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _resolve_model_pricing(model_name: str) -> dict[str, float] | None:
    normalized = model_name.strip().lower()
    if normalized in MODEL_PRICING_USD_PER_1M:
        return MODEL_PRICING_USD_PER_1M[normalized]

    for known_name, pricing in MODEL_PRICING_USD_PER_1M.items():
        if normalized.startswith(known_name):
            return pricing
    return None


def extract_retry_count(response_metadata: Any) -> int:
    if not isinstance(response_metadata, dict):
        return 0
    for key in ("retry_count", "retries", "num_retries"):
        value = response_metadata.get(key)
        if value is not None:
            return int(value)
    return 0


def extract_timeout_count(response_metadata: Any) -> int:
    if not isinstance(response_metadata, dict):
        return 0
    for key in ("timeout_count", "timeouts"):
        value = response_metadata.get(key)
        if value is not None:
            return int(value)
    if response_metadata.get("timed_out") is True:
        return 1
    return 0

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import yfinance as yf
from langchain_core.tools import tool

from .config import get_settings
from .retrieval import get_hybrid_retriever
from .schemas import EvidenceRecord, FundamentalsSnapshot

PROFILE_DOC_TYPES: dict[str, set[str]] = {
    "fundamentals": {"filing"},
    "sentiment": {"transcript", "news"},
    "risk": {"filing", "news", "transcript"},
}


def _iter_corpus_files(base_dir: Path) -> Iterable[Path]:
    for pattern in ("*.md", "*.txt", "*.json", "*.jsonl"):
        yield from base_dir.rglob(pattern)


def _parse_record(path: Path, ticker: str) -> list[EvidenceRecord]:
    if path.suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
        if ticker.upper() not in text.upper() and ticker.upper() not in path.name.upper():
            return []
        return [
            EvidenceRecord(
                source_id=path.stem,
                ticker=ticker,
                title=path.stem.replace("_", " "),
                content=text[:4000],
                document_type=_infer_doc_type(path),
                source_url=None,
                score=0.5,
            )
        ]

    records: list[EvidenceRecord] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    if path.suffix == ".json":
        payload = json.loads("\n".join(lines))
        payload = payload if isinstance(payload, list) else [payload]
    else:
        payload = [json.loads(line) for line in lines if line.strip()]

    for idx, item in enumerate(payload):
        item_ticker = str(item.get("ticker", "")).upper()
        if item_ticker and item_ticker != ticker.upper():
            continue
        content = str(item.get("content") or item.get("chunk_text") or "")
        if not content:
            continue
        records.append(
            EvidenceRecord(
                source_id=str(item.get("source_id", f"{path.stem}-{idx}")),
                ticker=ticker,
                title=str(item.get("title", path.stem)),
                content=content[:4000],
                document_type=_normalize_doc_type(
                    str(item.get("document_type", _infer_doc_type(path)))
                ),
                source_url=item.get("source_url"),
                published_at=_parse_datetime(item.get("published_at")),
                score=float(item.get("score", 0.5)),
                provider=item.get("provider"),
                section=item.get("section"),
                form_type=item.get("form_type"),
                document_id=item.get("document_id"),
                accession_number=item.get("accession_number"),
                chunk_index=item.get("chunk_index"),
                metadata_version=item.get("metadata_version"),
                speaker=item.get("speaker"),
                speaker_role=item.get("speaker_role"),
                publisher=item.get("publisher"),
                sentiment_label=item.get("sentiment_label"),
                sentiment_score=item.get("sentiment_score"),
                ticker_relevance_score=item.get("ticker_relevance_score"),
                entity_title_match=item.get("entity_title_match"),
                entity_body_match=item.get("entity_body_match"),
                news_relevance_score=item.get("news_relevance_score"),
                news_relevance_tier=item.get("news_relevance_tier"),
                source_quality_tier=item.get("source_quality_tier"),
            )
        )
    return records


def _parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _infer_doc_type(path: Path) -> str:
    name = path.name.lower()
    if "10-k" in name or "10q" in name or "filing" in name:
        return "filing"
    if "transcript" in name or "earnings" in name:
        return "transcript"
    if "news" in name:
        return "news"
    if "fundamental" in name:
        return "fundamentals"
    return "note"


def _normalize_doc_type(value: str) -> str:
    allowed = {"filing", "transcript", "news", "fundamentals", "note", "unknown"}
    normalized = value.lower().strip()
    return normalized if normalized in allowed else "unknown"


def _boost_for_profile(record: EvidenceRecord, profile: str | None) -> float:
    if profile is None:
        return 0.0

    boost = 0.0
    allowed = PROFILE_DOC_TYPES.get(profile, set())
    if record.document_type in allowed:
        boost += 3.0

    if profile == "fundamentals":
        if record.document_type == "filing" and record.section in {
            "item_1_business",
            "item_7_mda",
            "item_8_financial_statements",
            "part_i_item_2_mda",
        }:
            boost += 2.0
    elif profile == "sentiment":
        if record.document_type == "transcript":
            boost += 2.5
        if record.document_type == "news":
            boost += 1.5
            boost += _news_quality_boost(record)
        if record.speaker_role and any(
            token in record.speaker_role.lower()
            for token in ("chief", "ceo", "cfo", "president", "investor relations")
        ):
            boost += 1.0
        if record.sentiment_score is not None:
            boost += min(abs(record.sentiment_score), 1.0)
    elif profile == "risk":
        if record.document_type == "filing" and record.section in {
            "item_1a_risk_factors",
            "item_7a_market_risk",
            "part_ii_item_1a_risk_factors",
        }:
            boost += 3.0
        if record.document_type == "news":
            boost += 1.5
            boost += _news_quality_boost(record)
        if record.sentiment_label and record.sentiment_label.lower() in {
            "bearish",
            "somewhat-bearish",
        }:
            boost += 1.0
    return boost


def _news_quality_boost(record: EvidenceRecord) -> float:
    if record.document_type != "news":
        return 0.0
    score = float(record.news_relevance_score or 0.0) * 2.0
    if record.source_quality_tier == "trusted":
        score += 0.35
    elif record.source_quality_tier == "standard":
        score += 0.15
    else:
        score -= 0.15
    return score


def merge_evidence_sets(*collections: list[EvidenceRecord]) -> list[EvidenceRecord]:
    merged: dict[str, EvidenceRecord] = {}
    for records in collections:
        for record in records:
            existing = merged.get(record.source_id)
            if existing is None or record.score > existing.score:
                merged[record.source_id] = record
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)


def local_corpus_search(
    query: str,
    ticker: str,
    top_k: int | None = None,
    *,
    profile: str | None = None,
) -> list[EvidenceRecord]:
    settings = get_settings()
    if getattr(settings, "db_enabled", False):
        try:
            results = get_hybrid_retriever().search(
                query=query,
                ticker=ticker,
                top_k=top_k,
                profile=profile,
            )
            if results:
                return results
        except Exception:
            # Keep a local-file fallback so development workflows still function
            # when the DB is unavailable or the retrieval index is not populated yet.
            pass

    if not settings.corpus_dir.exists():
        return []

    top_k = top_k or settings.default_top_k
    scored: list[EvidenceRecord] = []
    query_terms = {term.lower() for term in query.split() if term.strip()}

    for path in _iter_corpus_files(settings.corpus_dir):
        for record in _parse_record(path, ticker):
            haystack = f"{record.title}\n{record.content}".lower()
            overlap = sum(term in haystack for term in query_terms)
            if overlap == 0 and ticker.lower() not in haystack:
                continue
            if profile and record.document_type not in PROFILE_DOC_TYPES.get(profile, set()):
                continue
            record.score = float(overlap) + _boost_for_profile(record, profile)
            scored.append(record)

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


@tool
def retrieve_corpus_evidence(
    query: str,
    ticker: str,
    top_k: int = 4,
    profile: str | None = None,
) -> list[dict]:
    """Retrieve local corpus evidence for a ticker and query."""
    results = local_corpus_search(query=query, ticker=ticker, top_k=top_k, profile=profile)
    return [record.model_dump(mode="json") for record in results]


def fetch_fundamentals_snapshot(ticker: str) -> FundamentalsSnapshot:
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    return FundamentalsSnapshot(
        ticker=ticker,
        as_of=datetime.utcnow(),
        metrics={
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "revenue_growth": info.get("revenueGrowth"),
            "return_on_equity": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "free_cash_flow": info.get("freeCashflow"),
            "operating_margins": info.get("operatingMargins"),
        },
    )


def fundamentals_snapshot_to_evidence(snapshot: FundamentalsSnapshot) -> list[EvidenceRecord]:
    metric_order = (
        "market_cap",
        "trailing_pe",
        "forward_pe",
        "peg_ratio",
        "revenue_growth",
        "return_on_equity",
        "debt_to_equity",
        "current_ratio",
        "free_cash_flow",
        "operating_margins",
    )
    as_of = snapshot.as_of.isoformat() if snapshot.as_of else "unknown"
    source_url = f"https://finance.yahoo.com/quote/{snapshot.ticker}"
    records: list[EvidenceRecord] = []

    for metric_name in metric_order:
        value = snapshot.metrics.get(metric_name)
        if value is None:
            continue
        rendered_value = value if isinstance(value, str) else repr(value)
        records.append(
            EvidenceRecord(
                source_id=f"{snapshot.ticker.lower()}-fundamentals-{metric_name}",
                ticker=snapshot.ticker,
                title=f"{snapshot.ticker} fundamentals: {metric_name}",
                content=(
                    f"Ticker: {snapshot.ticker}\n"
                    f"Metric: {metric_name}\n"
                    f"Value: {rendered_value}\n"
                    f"As of: {as_of}\n"
                    f"Provider: {snapshot.source}"
                ),
                document_type="fundamentals",
                source_url=source_url,
                published_at=snapshot.as_of,
                provider=snapshot.source,
                metadata_version="1.0",
            )
        )
    return records


@tool
def retrieve_fundamentals(ticker: str) -> dict:
    """Fetch a fundamentals snapshot for a ticker."""
    return fetch_fundamentals_snapshot(ticker).model_dump(mode="json")


def build_evidence_context(records: list[EvidenceRecord]) -> str:
    if not records:
        return "No evidence retrieved."
    chunks = []
    for record in records:
        chunks.append(
            f"[source:{record.source_id}] {record.document_type} | {record.title}\n"
            f"section={record.section} form_type={record.form_type} "
            f"speaker={record.speaker} speaker_role={record.speaker_role} "
            f"publisher={record.publisher} sentiment_label={record.sentiment_label} "
            f"sentiment_score={record.sentiment_score} "
            f"published_at={record.published_at} url={record.source_url}\n"
            f"{record.content[:1200]}"
        )
    return "\n\n".join(chunks)

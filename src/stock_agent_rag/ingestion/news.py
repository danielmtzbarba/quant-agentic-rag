from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from ..config import Settings, get_settings
from ..db import get_db_session, initialize_database
from ..logging import get_logger
from ..registry import RegistryService
from ..schemas import DocumentRecord, EvidenceChunk

logger = get_logger(__name__)

DEFAULT_TICKER_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple", "apple inc", "apple inc."),
    "AMD": ("advanced micro devices", "amd"),
    "AMZN": ("amazon", "amazon.com", "amazon.com inc"),
    "GOOG": ("google", "alphabet", "alphabet inc", "google llc"),
    "GOOGL": ("google", "alphabet", "alphabet inc", "google llc"),
    "INTC": ("intel", "intel corporation"),
    "META": ("meta", "meta platforms", "facebook"),
    "MSFT": ("microsoft", "microsoft corporation"),
    "NVDA": ("nvidia", "nvidia corporation"),
    "TSLA": ("tesla", "tesla inc", "tesla motors"),
}
TRUSTED_NEWS_SOURCES = {
    "associated press",
    "ap",
    "bloomberg",
    "cnbc",
    "financial times",
    "ft",
    "reuters",
    "the wall street journal",
    "wall street journal",
    "wsj",
}
STANDARD_NEWS_SOURCES = {
    "barron's",
    "benzinga",
    "fortune",
    "investor's business daily",
    "marketwatch",
    "motley fool",
    "seeking alpha",
    "the globe and mail",
    "tradingkey",
    "yahoo finance",
}


class AlphaVantageNewsArticle(BaseModel):
    title: str
    url: str
    time_published: str
    summary: str | None = None
    source: str | None = None
    overall_sentiment_score: str | float | None = None
    overall_sentiment_label: str | None = None
    ticker_sentiment: list[dict] = []

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate_required(self) -> AlphaVantageNewsArticle:
        self.title = self.title.strip()
        self.url = self.url.strip()
        self.time_published = self.time_published.strip()
        self.summary = (self.summary or "").strip() or None
        self.source = (self.source or "").strip() or None
        if not self.title:
            raise ValueError("title is required")
        if not self.url:
            raise ValueError("url is required")
        if not self.time_published:
            raise ValueError("time_published is required")
        return self


class AlphaVantageNewsCacheFile(BaseModel):
    provider: str | None = None
    fetched_at: datetime | None = None
    request_url: str | None = None
    http_status: int | None = None
    payload: dict[str, object]

    model_config = ConfigDict(extra="allow")


@dataclass(slots=True)
class NewsFetchResult:
    payload: dict[str, object]
    request_url: str
    http_status: int | None
    fetched_at: datetime


@dataclass(slots=True)
class IngestionSummary:
    ticker: str
    processed_documents: int
    chunk_count: int
    normalized_paths: list[str]
    chunk_paths: list[str]


@dataclass(frozen=True, slots=True)
class NewsRelevanceAssessment:
    ticker_relevance_score: float
    entity_title_match: bool
    entity_body_match: bool
    news_relevance_score: float
    news_relevance_tier: str
    source_quality_tier: str


class AlphaVantageNewsIngestionService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def ingest(self, ticker: str, limit: int = 20, force: bool = False) -> IngestionSummary:
        ticker = ticker.upper()
        self._validate_vantage_config()
        registry_service: RegistryService | None = None
        run_id: str | None = None
        if self.settings.db_enabled:
            initialize_database(self.settings)
            registry_service = RegistryService(get_db_session())
            run_id = registry_service.create_ingestion_run(
                source_type="news",
                ticker=ticker,
                form_type=None,
                metadata_version=self.settings.news_metadata_version,
            )

        normalized_paths: list[str] = []
        chunk_paths: list[str] = []
        chunk_total = 0

        try:
            raw_path = self._raw_payload_path(ticker=ticker, limit=limit)
            used_cache = False
            raw_payload: dict[str, object] | None = None

            if raw_path.exists() and not force:
                try:
                    raw_payload = self._load_cached_payload(raw_path)
                    used_cache = True
                except Exception as exc:
                    logger.warning(
                        "cached news payload invalid; refetching",
                        extra={
                            "ticker": ticker,
                            "cached_raw_path": str(raw_path),
                            "error": str(exc),
                        },
                    )

            if raw_payload is None:
                fetch_result = self._fetch_news_payload(ticker=ticker, limit=limit)
                raw_payload = fetch_result.payload
                raw_path_str = self._persist_raw_payload(
                    ticker=ticker,
                    limit=limit,
                    payload=raw_payload,
                    request_url=fetch_result.request_url,
                    http_status=fetch_result.http_status,
                    fetched_at=fetch_result.fetched_at,
                )
            else:
                raw_path_str = str(raw_path)

            articles = self._extract_articles(raw_payload, ticker=ticker)
            for article in articles:
                document = self._build_document_record(
                    article=article,
                    raw_path=raw_path_str,
                    ticker=ticker,
                )
                chunks = self._chunk_document(document)
                normalized_path = self._persist_document(document)
                chunk_path = self._persist_chunks(document=document, chunks=chunks)
                normalized_paths.append(normalized_path)
                chunk_paths.append(chunk_path)
                chunk_total += len(chunks)

                if registry_service is not None:
                    registry_service.upsert_document(
                        document=document,
                        normalized_path=normalized_path,
                    )
                    registry_service.upsert_chunks(chunks=chunks, chunk_path=chunk_path)

            if registry_service is not None and run_id is not None:
                registry_service.complete_ingestion_run(
                    run_id=run_id,
                    processed_documents=len(articles),
                    chunk_count=chunk_total,
                )
        except Exception as exc:
            if registry_service is not None and run_id is not None:
                registry_service.complete_ingestion_run(
                    run_id=run_id,
                    processed_documents=len(normalized_paths),
                    chunk_count=chunk_total,
                    error_message=str(exc),
                )
            raise

        logger.info(
            "news batch processed",
            extra={
                "ticker": ticker,
                "used_cache": used_cache,
                "documents": len(normalized_paths),
                "chunks": chunk_total,
            },
        )

        return IngestionSummary(
            ticker=ticker,
            processed_documents=len(normalized_paths),
            chunk_count=chunk_total,
            normalized_paths=normalized_paths,
            chunk_paths=chunk_paths,
        )

    def _raw_payload_path(self, *, ticker: str, limit: int) -> Path:
        return self.settings.news_raw_dir / "alpha_vantage" / ticker / f"latest-limit-{limit}.json"

    def _load_cached_payload(self, raw_path: Path) -> dict[str, object]:
        payload_json = json.loads(raw_path.read_text(encoding="utf-8"))
        if isinstance(payload_json, dict) and "payload" in payload_json:
            cache_file = AlphaVantageNewsCacheFile.model_validate(payload_json)
            return cache_file.payload
        if not isinstance(payload_json, dict):
            raise RuntimeError("Cached news payload must be a JSON object.")
        return payload_json

    def _fetch_news_payload(self, ticker: str, limit: int) -> NewsFetchResult:
        params = urlencode(
            {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "limit": limit,
                "apikey": self.settings.vantage_api_key,
            }
        )
        url = f"{self.settings.vantage_base_url.rstrip('?')}?{params}"
        request = Request(
            url,
            headers={
                "accept": "application/json",
                "user-agent": f"{self.settings.app_name}/{self.settings.app_env}",
            },
        )

        max_attempts = max(1, int(self.settings.transcript_http_max_retries))
        timeout_s = float(self.settings.transcript_http_timeout_s)
        backoff_base_s = float(self.settings.transcript_http_backoff_base_s)
        backoff_max_s = float(self.settings.transcript_http_backoff_max_s)
        retryable_statuses = {429, 500, 502, 503, 504}
        last_exc: Exception | None = None

        for attempt in range(max_attempts):
            try:
                with urlopen(request, timeout=timeout_s) as response:
                    status = getattr(response, "status", None) or response.getcode()
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise RuntimeError("Unexpected news response format from Alpha Vantage.")
                if "Error Message" in payload:
                    raise RuntimeError(str(payload["Error Message"]))
                if "Information" in payload and "feed" not in payload:
                    raise RuntimeError(str(payload["Information"]))
                if "Note" in payload and "feed" not in payload:
                    raise RuntimeError(str(payload["Note"]))
                return NewsFetchResult(
                    payload=payload,
                    request_url=url,
                    http_status=int(status) if status is not None else None,
                    fetched_at=datetime.now(UTC),
                )
            except HTTPError as exc:
                status = int(getattr(exc, "code", -1))
                if status == 401:
                    raise RuntimeError(
                        "Alpha Vantage request was rejected. Check VANTAGE_API_KEY in .env."
                    ) from exc
                if status == 403:
                    raise RuntimeError(
                        "Alpha Vantage news request was forbidden (HTTP 403). "
                        "Verify `VANTAGE_API_KEY` in .env and your account permissions."
                    ) from exc
                last_exc = exc
                if status in retryable_statuses and attempt < max_attempts - 1:
                    sleep(
                        min(backoff_max_s, backoff_base_s * (2**attempt))
                        * (1.0 + random.random() * 0.2)
                    )
                    continue
                raise RuntimeError(
                    f"Alpha Vantage news request failed with HTTP {status}."
                ) from exc
            except URLError as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    sleep(
                        min(backoff_max_s, backoff_base_s * (2**attempt))
                        * (1.0 + random.random() * 0.2)
                    )
                    continue
                raise RuntimeError(
                    "Alpha Vantage news request failed due to network connectivity."
                ) from exc
            except json.JSONDecodeError as exc:
                raise RuntimeError("Alpha Vantage news response was not valid JSON.") from exc

        if last_exc is not None:
            raise RuntimeError("Alpha Vantage news request failed after retries.") from last_exc
        raise RuntimeError("Alpha Vantage news request failed after retries with unknown error.")

    def _extract_articles(
        self,
        raw_payload: dict[str, object],
        *,
        ticker: str,
    ) -> list[AlphaVantageNewsArticle]:
        feed = raw_payload.get("feed")
        if not isinstance(feed, list):
            raise RuntimeError("Alpha Vantage news payload did not include a feed list.")

        articles: list[AlphaVantageNewsArticle] = []
        for item in feed:
            try:
                article = AlphaVantageNewsArticle.model_validate(item)
            except ValidationError as exc:
                logger.warning(
                    "skipping invalid news article",
                    extra={"ticker": ticker, "error": str(exc)},
                )
                continue
            assessment = self._assess_article_relevance(article=article, ticker=ticker)
            if assessment is None:
                continue
            articles.append(article)

        if not articles:
            raise RuntimeError(f"Alpha Vantage returned no news articles for {ticker}.")
        return articles

    def _assess_article_relevance(
        self,
        *,
        article: AlphaVantageNewsArticle,
        ticker: str,
    ) -> NewsRelevanceAssessment | None:
        title_text = article.title.lower()
        body_text = f"{article.title}\n{article.summary or ''}".lower()
        aliases = self._entity_aliases(ticker)
        entity_title_match = any(self._contains_alias(title_text, alias) for alias in aliases)
        entity_body_match = any(self._contains_alias(body_text, alias) for alias in aliases)
        ticker_relevance_score = self._ticker_relevance_score(article=article, ticker=ticker)
        source_quality_tier = self._source_quality_tier(article.source)

        if not entity_title_match and not entity_body_match:
            logger.info(
                "skipping off-ticker news article",
                extra={
                    "ticker": ticker,
                    "title": article.title,
                    "publisher": article.source,
                    "ticker_relevance_score": ticker_relevance_score,
                },
            )
            return None

        news_relevance_score = min(
            1.0,
            (0.65 if entity_title_match else 0.0)
            + (0.25 if entity_body_match else 0.0)
            + 0.10 * ticker_relevance_score,
        )
        if entity_title_match:
            news_relevance_tier = "direct"
        elif entity_body_match:
            news_relevance_tier = "body_only"
        else:
            news_relevance_tier = "indirect"

        return NewsRelevanceAssessment(
            ticker_relevance_score=ticker_relevance_score,
            entity_title_match=entity_title_match,
            entity_body_match=entity_body_match,
            news_relevance_score=news_relevance_score,
            news_relevance_tier=news_relevance_tier,
            source_quality_tier=source_quality_tier,
        )

    def _build_document_record(
        self,
        *,
        article: AlphaVantageNewsArticle,
        raw_path: str,
        ticker: str,
    ) -> DocumentRecord:
        published_at = self._parse_datetime(article.time_published)
        title_hash = self._checksum(f"{ticker}:{article.url}")[:12]
        document_id = f"{ticker.lower()}-news-{title_hash}"
        sentiment_score = self._to_float(article.overall_sentiment_score)
        assessment = self._assess_article_relevance(article=article, ticker=ticker)
        if assessment is None:
            raise RuntimeError(
                "news relevance validation failed for article "
                f"`{article.title}` and ticker {ticker}"
            )
        cleaned_text = self._build_article_text(article=article, assessment=assessment)

        return DocumentRecord(
            document_id=document_id,
            source_type="news",
            ticker=ticker,
            title=article.title,
            provider="alpha_vantage",
            published_at=published_at,
            as_of_date=published_at.date() if published_at else None,
            source_url=article.url,
            metadata_version=self.settings.news_metadata_version,
            raw_checksum=self._checksum(
                json.dumps(article.model_dump(mode="json"), sort_keys=True)
            ),
            raw_path=raw_path,
            cleaned_text=cleaned_text,
            publisher=article.source,
            sentiment_label=article.overall_sentiment_label,
            sentiment_score=sentiment_score,
            ticker_relevance_score=assessment.ticker_relevance_score,
            entity_title_match=assessment.entity_title_match,
            entity_body_match=assessment.entity_body_match,
            news_relevance_score=assessment.news_relevance_score,
            news_relevance_tier=assessment.news_relevance_tier,
            source_quality_tier=assessment.source_quality_tier,
        )

    def _chunk_document(self, document: DocumentRecord) -> list[EvidenceChunk]:
        return [
            EvidenceChunk(
                chunk_id=f"{document.document_id}-000",
                source_id=f"{document.document_id}-000",
                document_id=document.document_id,
                ticker=document.ticker,
                title=document.title,
                content=document.cleaned_text,
                document_type="news",
                provider=document.provider,
                source_url=document.source_url,
                published_at=document.published_at,
                chunk_index=0,
                metadata_version=document.metadata_version,
                publisher=document.publisher,
                sentiment_label=document.sentiment_label,
                sentiment_score=document.sentiment_score,
                ticker_relevance_score=document.ticker_relevance_score,
                entity_title_match=document.entity_title_match,
                entity_body_match=document.entity_body_match,
                news_relevance_score=document.news_relevance_score,
                news_relevance_tier=document.news_relevance_tier,
                source_quality_tier=document.source_quality_tier,
            )
        ]

    def _persist_raw_payload(
        self,
        *,
        ticker: str,
        limit: int,
        payload: dict[str, object],
        request_url: str,
        http_status: int | None,
        fetched_at: datetime,
    ) -> str:
        base_dir = self.settings.news_raw_dir / "alpha_vantage" / ticker
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"latest-limit-{limit}.json"
        cache_payload = {
            "provider": "alpha_vantage",
            "fetched_at": fetched_at.isoformat(),
            "request_url": request_url,
            "http_status": http_status,
            "ticker": ticker,
            "limit": limit,
            "payload": payload,
        }
        output_path.write_text(
            json.dumps(cache_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return str(output_path)

    def _persist_document(self, document: DocumentRecord) -> str:
        base_dir = self.settings.normalized_data_dir / "news" / document.ticker
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{document.document_id}.json"
        output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return str(output_path)

    def _persist_chunks(self, document: DocumentRecord, chunks: list[EvidenceChunk]) -> str:
        base_dir = self.settings.chunked_data_dir / "news" / document.ticker
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{document.document_id}.jsonl"
        output_path.write_text(
            "\n".join(chunk.model_dump_json() for chunk in chunks),
            encoding="utf-8",
        )
        return str(output_path)

    def _build_article_text(
        self,
        *,
        article: AlphaVantageNewsArticle,
        assessment: NewsRelevanceAssessment | None = None,
    ) -> str:
        parts = [f"Headline: {article.title}"]
        if article.source:
            parts.append(f"Publisher: {article.source}")
        if article.overall_sentiment_label:
            parts.append(f"Sentiment label: {article.overall_sentiment_label}")
        score = self._to_float(article.overall_sentiment_score)
        if score is not None:
            parts.append(f"Sentiment score: {score}")
        if article.summary:
            parts.append(f"Summary: {article.summary}")
        if assessment is not None:
            parts.append(f"Ticker relevance score: {assessment.ticker_relevance_score}")
            parts.append(f"Entity title match: {assessment.entity_title_match}")
            parts.append(f"Entity body match: {assessment.entity_body_match}")
            parts.append(f"News relevance score: {assessment.news_relevance_score}")
            parts.append(f"News relevance tier: {assessment.news_relevance_tier}")
            parts.append(f"Source quality tier: {assessment.source_quality_tier}")
        return "\n".join(parts)

    def _entity_aliases(self, ticker: str) -> tuple[str, ...]:
        aliases = [ticker.lower(), f"${ticker.lower()}"]
        aliases.extend(DEFAULT_TICKER_ENTITY_ALIASES.get(ticker.upper(), ()))
        # Preserve order while removing duplicates and empty entries.
        deduped: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            normalized = alias.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return tuple(deduped)

    def _contains_alias(self, text: str, alias: str) -> bool:
        alias = alias.strip().lower()
        if not alias:
            return False
        if " " in alias or "." in alias:
            return alias in text
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", text) is not None

    def _ticker_relevance_score(self, *, article: AlphaVantageNewsArticle, ticker: str) -> float:
        best = 0.0
        for sentiment in article.ticker_sentiment:
            if not isinstance(sentiment, dict):
                continue
            if str(sentiment.get("ticker", "")).upper() != ticker.upper():
                continue
            best = max(best, self._to_float(sentiment.get("relevance_score")) or 0.0)
        return min(max(best, 0.0), 1.0)

    def _source_quality_tier(self, publisher: str | None) -> str:
        normalized = (publisher or "").strip().lower()
        if normalized in TRUSTED_NEWS_SOURCES:
            return "trusted"
        if normalized in STANDARD_NEWS_SOURCES:
            return "standard"
        return "low"

    def _parse_datetime(self, value: object) -> datetime | None:
        if not value:
            return None
        for fmt in ("%Y%m%dT%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(value), fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        return None

    def _to_float(self, value: object) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _checksum(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _validate_vantage_config(self) -> None:
        api_key = (self.settings.vantage_api_key or "").strip()
        invalid_keys = {
            "",
            "your_key",
            "your-api-key",
            "your_api_key",
            "yourapikey",
            "changeme",
            "your_vantage_key",
            "your vantage key",
            "demo",
        }
        if api_key.lower() in invalid_keys:
            raise ValueError("VANTAGE_API_KEY must be configured with a real Alpha Vantage key.")

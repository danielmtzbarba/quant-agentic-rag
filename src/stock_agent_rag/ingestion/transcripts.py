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

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from ..config import Settings, get_settings
from ..db import get_db_session, initialize_database
from ..logging import get_logger
from ..registry import RegistryService
from ..schemas import DocumentRecord, EvidenceChunk, TranscriptTurn

logger = get_logger(__name__)

SPEAKER_HEADER_RE = re.compile(
    r"^(?P<speaker>[A-Z][A-Za-z0-9 .,'&()/+-]{1,120}?)"
    r"(?:\s*[-\u2013\u2014]{1,2}\s*(?P<role>[^:]{2,140}))?:$"
)
PARTICIPANT_ROLE_RE = re.compile(
    r"^(?P<speaker>[A-Z][A-Za-z0-9 .,'&()/+-]{1,120}?)"
    r"\s*[-\u2013\u2014]{1,2}\s*(?P<role>[A-Za-z][A-Za-z0-9 ,.&()/+-]{2,140})$"
)
QUESTIONER_HINTS = ("analyst", "question", "q&a", "qa")
EXECUTIVE_HINTS = (
    "chief",
    "ceo",
    "cfo",
    "coo",
    "president",
    "vice president",
    "svp",
    "evp",
    "director",
    "investor relations",
)


class AlphaVantageTranscriptPayload(BaseModel):
    """Validated subset of the Alpha Vantage earnings transcript payload."""

    symbol: str | None = None
    quarter: str
    date: str | None = None
    transcript: object | None = None
    content: str | None = None
    title: str | None = None
    url: str | None = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _normalize_payload(self) -> AlphaVantageTranscriptPayload:
        transcript_text = self._coerce_transcript_text(self.transcript or self.content)
        if not transcript_text:
            raise ValueError("transcript content is required")
        self.transcript = transcript_text
        self.content = transcript_text
        self.quarter = self.quarter.strip()
        if not self.quarter:
            raise ValueError("quarter is required")
        if self.date is not None:
            self.date = self.date.strip() or None
        return self

    def _coerce_transcript_text(self, value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            rendered_parts: list[str] = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                speaker = str(item.get("speaker", "")).strip() or "Unknown Speaker"
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                if role:
                    rendered_parts.append(f"{speaker} -- {role}\n{speaker}:\n{content}")
                else:
                    rendered_parts.append(f"{speaker}:\n{content}")
            return "\n\n".join(rendered_parts).strip()
        return ""


class AlphaVantageTranscriptCacheFile(BaseModel):
    """On-disk cache format for transcript payload + fetch metadata."""

    provider: str | None = None
    fetched_at: datetime | None = None
    request_url: str | None = None
    http_status: int | None = None
    payload: AlphaVantageTranscriptPayload

    model_config = ConfigDict(extra="allow")


@dataclass(slots=True)
class TranscriptFetchResult:
    payload: dict[str, object]
    request_url: str
    http_status: int | None
    fetched_at: datetime


@dataclass(slots=True)
class IngestionSummary:
    ticker: str
    year: int
    quarter: int
    processed_documents: int
    chunk_count: int
    normalized_paths: list[str]
    chunk_paths: list[str]


class AlphaVantageTranscriptIngestionService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=3200,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def ingest(self, ticker: str, year: int, quarter: int, force: bool = False) -> IngestionSummary:
        ticker = ticker.upper()
        self._validate_vantage_config()
        registry_service: RegistryService | None = None
        run_id: str | None = None
        if self.settings.db_enabled:
            initialize_database(self.settings)
            registry_service = RegistryService(get_db_session())
            run_id = registry_service.create_ingestion_run(
                source_type="transcript",
                ticker=ticker,
                form_type=f"Q{quarter}-{year}",
                metadata_version=self.settings.transcript_metadata_version,
            )

        try:
            raw_path = self._raw_payload_path(ticker=ticker, year=year, quarter=quarter)
            used_cache = False
            raw_payload: dict[str, object] | None = None

            if raw_path.exists() and not force:
                try:
                    raw_payload = self._load_cached_payload(raw_path)
                    used_cache = True
                except Exception as exc:
                    logger.warning(
                        "cached transcript payload invalid; refetching",
                        extra={
                            "ticker": ticker,
                            "year": year,
                            "quarter": quarter,
                            "cached_raw_path": str(raw_path),
                            "error": str(exc),
                        },
                    )

            if raw_payload is None:
                fetch_result = self._fetch_transcript_payload(
                    ticker=ticker,
                    year=year,
                    quarter=quarter,
                )
                raw_payload = fetch_result.payload
                raw_path_str = self._persist_raw_payload(
                    ticker=ticker,
                    year=year,
                    quarter=quarter,
                    payload=raw_payload,
                    request_url=fetch_result.request_url,
                    http_status=fetch_result.http_status,
                    fetched_at=fetch_result.fetched_at,
                )
            else:
                raw_path_str = str(raw_path)

            document = self._build_document_record(
                raw_payload=raw_payload,
                raw_path=raw_path_str,
                ticker=ticker,
                year=year,
                quarter=quarter,
            )
            chunks = self._chunk_document(document)
            normalized_path = self._persist_document(document)
            chunk_path = self._persist_chunks(document=document, chunks=chunks)

            if registry_service is not None and run_id is not None:
                registry_service.upsert_document(
                    document=document,
                    normalized_path=normalized_path,
                )
                registry_service.upsert_chunks(chunks=chunks, chunk_path=chunk_path)
                registry_service.complete_ingestion_run(
                    run_id=run_id,
                    processed_documents=1,
                    chunk_count=len(chunks),
                )
        except Exception as exc:
            if registry_service is not None and run_id is not None:
                registry_service.complete_ingestion_run(
                    run_id=run_id,
                    processed_documents=0,
                    chunk_count=0,
                    error_message=str(exc),
                )
            raise

        logger.info(
            "transcript processed",
            extra={
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "used_cache": used_cache,
                "document_id": document.document_id,
                "turns": len(document.transcript_turns),
                "chunks": len(chunks),
            },
        )

        return IngestionSummary(
            ticker=ticker,
            year=year,
            quarter=quarter,
            processed_documents=1,
            chunk_count=len(chunks),
            normalized_paths=[normalized_path],
            chunk_paths=[chunk_path],
        )

    def extract_transcript_turns(self, transcript_text: str) -> list[TranscriptTurn]:
        normalized = self._normalize_text(transcript_text)
        if not normalized:
            return []

        lines = [line.strip() for line in normalized.splitlines()]
        participant_roles = self._extract_participant_roles(lines)

        turns: list[TranscriptTurn] = []
        current_speaker: str | None = None
        current_role: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            if not current_speaker or not buffer:
                buffer = []
                return
            content = self._normalize_text("\n".join(buffer))
            if len(content) < 20:
                buffer = []
                return
            turns.append(
                TranscriptTurn(
                    turn_id=f"turn-{len(turns):03d}",
                    speaker=current_speaker,
                    speaker_role=self._infer_speaker_role(
                        current_role or participant_roles.get(current_speaker),
                        content,
                    ),
                    content=content,
                    order=len(turns),
                )
            )
            buffer = []

        for line in lines:
            if not line:
                if current_speaker and buffer:
                    buffer.append("")
                continue

            header = self._parse_speaker_header(line)
            if header is not None:
                flush()
                current_speaker, explicit_role = header
                current_role = explicit_role or participant_roles.get(current_speaker)
                continue

            if current_speaker is None:
                continue
            buffer.append(line)

        flush()
        if turns:
            return turns

        return [
            TranscriptTurn(
                turn_id="turn-000",
                speaker="Unknown Speaker",
                speaker_role=None,
                content=normalized,
                order=0,
            )
        ]

    def _raw_payload_path(self, *, ticker: str, year: int, quarter: int) -> Path:
        return (
            self.settings.transcript_raw_dir
            / "alpha_vantage"
            / ticker
            / str(year)
            / f"q{quarter}.json"
        )

    def _load_cached_payload(self, raw_path: Path) -> dict[str, object]:
        payload_json = json.loads(raw_path.read_text(encoding="utf-8"))
        if isinstance(payload_json, dict) and "payload" in payload_json:
            cache_file = AlphaVantageTranscriptCacheFile.model_validate(payload_json)
        else:
            cache_file = AlphaVantageTranscriptCacheFile.model_validate({"payload": payload_json})
        return cache_file.payload.model_dump(mode="python")

    def _validate_transcript_payload(
        self,
        record: object,
    ) -> AlphaVantageTranscriptPayload:
        if not isinstance(record, dict):
            raise RuntimeError(
                "Alpha Vantage transcript payload has unexpected format (expected object)."
            )
        try:
            return AlphaVantageTranscriptPayload.model_validate(record)
        except ValidationError as exc:
            raise RuntimeError(
                f"Alpha Vantage transcript payload validation failed: {exc}"
            ) from exc

    def _fetch_transcript_payload(
        self,
        ticker: str,
        year: int,
        quarter: int,
    ) -> TranscriptFetchResult:
        quarter_id = f"{year}Q{quarter}"
        params = urlencode(
            {
                "function": "EARNINGS_CALL_TRANSCRIPT",
                "symbol": ticker,
                "quarter": quarter_id,
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
                    raise RuntimeError("Unexpected transcript response format from Alpha Vantage.")
                if "Error Message" in payload:
                    raise RuntimeError(str(payload["Error Message"]))
                if (
                    "Information" in payload
                    and "transcript" not in payload
                    and "content" not in payload
                ):
                    raise RuntimeError(str(payload["Information"]))
                if "Note" in payload and "transcript" not in payload and "content" not in payload:
                    raise RuntimeError(str(payload["Note"]))

                validated = self._validate_transcript_payload(payload)
                return TranscriptFetchResult(
                    payload=validated.model_dump(mode="python"),
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
                if status == 404:
                    raise RuntimeError(
                        f"Alpha Vantage transcript not found for {ticker} {quarter_id}."
                    ) from exc
                if status == 403:
                    raise RuntimeError(
                        "Alpha Vantage request was forbidden (HTTP 403). "
                        "Verify `VANTAGE_API_KEY` in .env and your "
                        "Alpha Vantage account permissions."
                    ) from exc

                last_exc = exc
                if status in retryable_statuses and attempt < max_attempts - 1:
                    retry_after_s: float | None = None
                    retry_after_header = getattr(
                        getattr(exc, "headers", None),
                        "get",
                        lambda _key: None,
                    )("Retry-After")
                    if retry_after_header:
                        try:
                            retry_after_s = float(str(retry_after_header).strip())
                        except ValueError:
                            retry_after_s = None
                    if retry_after_s is None:
                        retry_after_s = min(backoff_max_s, backoff_base_s * (2**attempt))
                    sleep(retry_after_s * (1.0 + (random.random() * 0.2)))
                    continue
                raise RuntimeError(f"Alpha Vantage request failed with HTTP {status}.") from exc
            except URLError as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    retry_after_s = min(backoff_max_s, backoff_base_s * (2**attempt))
                    sleep(retry_after_s * (1.0 + (random.random() * 0.2)))
                    continue
                raise RuntimeError(
                    "Alpha Vantage transcript request failed due to network connectivity."
                ) from exc
            except json.JSONDecodeError as exc:
                raise RuntimeError("Alpha Vantage transcript response was not valid JSON.") from exc

        if last_exc is not None:
            raise RuntimeError(
                "Alpha Vantage transcript request failed after retries."
            ) from last_exc
        raise RuntimeError(
            "Alpha Vantage transcript request failed after retries with unknown error."
        )

    def _build_document_record(
        self,
        *,
        raw_payload: dict,
        raw_path: str,
        ticker: str,
        year: int,
        quarter: int,
    ) -> DocumentRecord:
        transcript_turns = self._extract_turns_from_payload(raw_payload)
        transcript_text = self._normalize_text(
            "\n\n".join(
                f"{turn.speaker}:\n{turn.content}" for turn in transcript_turns
            )
            if transcript_turns
            else str(raw_payload.get("transcript") or raw_payload.get("content") or "")
        )
        published_at = self._parse_datetime(raw_payload.get("date"))
        quarter_id = str(raw_payload.get("quarter") or f"{year}Q{quarter}")
        title = str(raw_payload.get("title") or f"{ticker} {quarter_id} earnings transcript")
        document_id = f"{ticker.lower()}-q{quarter}-{year}-transcript"

        return DocumentRecord(
            document_id=document_id,
            source_type="transcript",
            ticker=ticker,
            title=title,
            provider="alpha_vantage",
            form_type=f"Q{quarter}",
            published_at=published_at,
            as_of_date=published_at.date() if published_at else None,
            source_url=raw_payload.get("url"),
            metadata_version=self.settings.transcript_metadata_version,
            raw_checksum=self._checksum(json.dumps(raw_payload, sort_keys=True)),
            raw_path=raw_path,
            cleaned_text=transcript_text,
            transcript_turns=transcript_turns,
        )

    def _extract_turns_from_payload(self, raw_payload: dict) -> list[TranscriptTurn]:
        transcript_value = raw_payload.get("transcript")
        if not isinstance(transcript_value, list):
            transcript_text = self._normalize_text(
                str(transcript_value or raw_payload.get("content") or "")
            )
            return self.extract_transcript_turns(transcript_text)

        turns: list[TranscriptTurn] = []
        for item in transcript_value:
            if not isinstance(item, dict):
                continue
            speaker = self._clean_speaker(str(item.get("speaker", "") or "Unknown Speaker"))
            content = self._normalize_text(str(item.get("content", "")))
            if not content:
                continue
            role = self._clean_role(str(item.get("role", "")) or None)
            turns.append(
                TranscriptTurn(
                    turn_id=f"turn-{len(turns):03d}",
                    speaker=speaker or "Unknown Speaker",
                    speaker_role=self._infer_speaker_role(role, content),
                    content=content,
                    order=len(turns),
                )
            )

        if turns:
            return turns

        transcript_text = self._normalize_text(str(raw_payload.get("content") or ""))
        return self.extract_transcript_turns(transcript_text)

    def _chunk_document(self, document: DocumentRecord) -> list[EvidenceChunk]:
        turns = document.transcript_turns or [
            TranscriptTurn(
                turn_id="turn-000",
                speaker="Unknown Speaker",
                speaker_role=None,
                content=document.cleaned_text,
                order=0,
            )
        ]
        chunks: list[EvidenceChunk] = []
        for turn in turns:
            prefix = self._speaker_prefix(turn)
            segment_chunks = self._splitter.split_text(f"{prefix}\n{turn.content}".strip())
            for idx, chunk_text in enumerate(segment_chunks):
                chunk_id = f"{document.document_id}-{turn.turn_id}-{idx:03d}"
                chunks.append(
                    EvidenceChunk(
                        chunk_id=chunk_id,
                        source_id=chunk_id,
                        document_id=document.document_id,
                        ticker=document.ticker,
                        title=f"{document.title} | {turn.speaker}",
                        content=chunk_text,
                        document_type="transcript",
                        provider=document.provider,
                        form_type=document.form_type,
                        section=turn.turn_id,
                        source_url=document.source_url,
                        published_at=document.published_at,
                        chunk_index=idx,
                        metadata_version=document.metadata_version,
                        speaker=turn.speaker,
                        speaker_role=turn.speaker_role,
                    )
                )
        return chunks

    def _persist_raw_payload(
        self,
        *,
        ticker: str,
        year: int,
        quarter: int,
        payload: dict[str, object],
        request_url: str,
        http_status: int | None,
        fetched_at: datetime,
    ) -> str:
        base_dir = self.settings.transcript_raw_dir / "alpha_vantage" / ticker / str(year)
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"q{quarter}.json"
        cache_payload = {
            "provider": "alpha_vantage",
            "fetched_at": fetched_at.isoformat(),
            "request_url": request_url,
            "http_status": http_status,
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
            "payload": payload,
        }
        output_path.write_text(
            json.dumps(cache_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return str(output_path)

    def _persist_document(self, document: DocumentRecord) -> str:
        base_dir = self.settings.normalized_data_dir / "transcripts" / document.ticker
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{document.document_id}.json"
        output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return str(output_path)

    def _persist_chunks(self, document: DocumentRecord, chunks: list[EvidenceChunk]) -> str:
        base_dir = self.settings.chunked_data_dir / "transcripts" / document.ticker
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{document.document_id}.jsonl"
        output_path.write_text(
            "\n".join(chunk.model_dump_json() for chunk in chunks),
            encoding="utf-8",
        )
        return str(output_path)

    def _extract_participant_roles(self, lines: list[str]) -> dict[str, str]:
        roles: dict[str, str] = {}
        for line in lines:
            match = PARTICIPANT_ROLE_RE.match(line)
            if not match:
                continue
            speaker = self._clean_speaker(match.group("speaker"))
            role = self._clean_role(match.group("role"))
            if speaker and role:
                roles[speaker] = role
        return roles

    def _parse_speaker_header(self, line: str) -> tuple[str, str | None] | None:
        match = SPEAKER_HEADER_RE.match(line)
        if not match:
            return None
        speaker = self._clean_speaker(match.group("speaker"))
        role = self._clean_role(match.group("role"))
        if not speaker:
            return None
        return speaker, role

    def _speaker_prefix(self, turn: TranscriptTurn) -> str:
        if turn.speaker_role:
            return f"Speaker: {turn.speaker} ({turn.speaker_role})"
        return f"Speaker: {turn.speaker}"

    def _clean_speaker(self, value: str | None) -> str:
        if not value:
            return ""
        return re.sub(r"\s+", " ", value).strip(" :-")

    def _clean_role(self, value: str | None) -> str | None:
        if not value:
            return None
        role = re.sub(r"\s+", " ", value).strip(" :-")
        return role or None

    def _parse_datetime(self, value: object) -> datetime | None:
        if not value:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(value), fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        return None

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

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

    def _infer_speaker_role(self, role: str | None, content: str) -> str | None:
        if role:
            return role
        lowered = content.lower()
        if any(token in lowered for token in QUESTIONER_HINTS):
            return "Analyst"
        if any(token in lowered for token in EXECUTIVE_HINTS):
            return "Executive"
        return None


FmpTranscriptIngestionService = AlphaVantageTranscriptIngestionService
FmpFetchResult = TranscriptFetchResult

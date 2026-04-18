from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from requests import HTTPError
from sec_edgar_downloader import Downloader

from ..config import Settings, get_settings
from ..db import get_db_session, initialize_database
from ..logging import get_logger
from ..registry import RegistryService
from ..schemas import DocumentRecord, EvidenceChunk, FilingSection

logger = get_logger(__name__)

TARGET_SECTIONS: dict[str, tuple[str, ...]] = {
    "10-K": ("1", "1A", "7", "7A", "8"),
    "10-Q": ("2", "1A"),
    "8-K": (),
}

SECTION_KEYS: dict[tuple[str, str, str | None], str] = {
    ("10-K", "1", None): "item_1_business",
    ("10-K", "1A", None): "item_1a_risk_factors",
    ("10-K", "7", None): "item_7_mda",
    ("10-K", "7A", None): "item_7a_market_risk",
    ("10-K", "8", None): "item_8_financial_statements",
    ("10-Q", "2", "part_i"): "part_i_item_2_mda",
    ("10-Q", "1A", "part_ii"): "part_ii_item_1a_risk_factors",
}

ITEM_HEADER_RE = re.compile(
    r"(?im)(^|\n)\s*(?:(part\s+(i{1,3}|iv))\s+)?item\s+(\d{1,2}[a-z]?)\s*[\.\-:]?\s+([^\n]{1,180})"
)
ACCESSION_RE = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
FILED_AS_OF_RE = re.compile(r"filed\s+as\s+of\s+date:\s*(\d{8})", re.IGNORECASE)
CIK_RE = re.compile(r"central\s+index\s+key:\s*(\d+)", re.IGNORECASE)


@dataclass(slots=True)
class IngestionSummary:
    ticker: str
    form_type: str
    processed_documents: int
    chunk_count: int
    normalized_paths: list[str]
    chunk_paths: list[str]


class SecFilingIngestionService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # Approximate ~1,000-token windows with character-based splitting so
        # ingestion stays deterministic and offline-safe in local dev and CI.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=600,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def ingest(self, ticker: str, form_type: str, limit: int = 1) -> IngestionSummary:
        form_type = form_type.upper()
        ticker = ticker.upper()
        self._validate_sec_identity()
        registry_service: RegistryService | None = None
        run_id: str | None = None
        if self.settings.db_enabled:
            initialize_database(self.settings)
            registry_service = RegistryService(get_db_session())
            run_id = registry_service.create_ingestion_run(
                source_type="filing",
                ticker=ticker,
                form_type=form_type,
                metadata_version=self.settings.sec_metadata_version,
            )

        self._download_filings(ticker=ticker, form_type=form_type, limit=limit)

        normalized_paths: list[str] = []
        chunk_paths: list[str] = []
        chunk_total = 0
        documents = self._discover_filing_documents(
            ticker=ticker,
            form_type=form_type,
            limit=limit,
        )

        try:
            for path in documents:
                document = self._build_document_record(
                    path=path,
                    ticker=ticker,
                    form_type=form_type,
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

                logger.info(
                    "sec filing processed",
                    extra={
                        "ticker": ticker,
                        "form_type": form_type,
                        "document_id": document.document_id,
                        "sections": len(document.sections),
                        "chunks": len(chunks),
                    },
                )

            if registry_service is not None and run_id is not None:
                registry_service.complete_ingestion_run(
                    run_id=run_id,
                    processed_documents=len(documents),
                    chunk_count=chunk_total,
                )
        except Exception as exc:
            if registry_service is not None and run_id is not None:
                registry_service.complete_ingestion_run(
                    run_id=run_id,
                    processed_documents=len(documents),
                    chunk_count=chunk_total,
                    error_message=str(exc),
                )
            raise

        return IngestionSummary(
            ticker=ticker,
            form_type=form_type,
            processed_documents=len(documents),
            chunk_count=chunk_total,
            normalized_paths=normalized_paths,
            chunk_paths=chunk_paths,
        )

    def extract_sections(self, text: str, form_type: str) -> list[FilingSection]:
        normalized_text = self._normalize_text(text)
        matches = list(ITEM_HEADER_RE.finditer(normalized_text))
        if not matches:
            return []

        sections: list[FilingSection] = []
        for idx, match in enumerate(matches):
            part_token = match.group(2)
            roman_part = match.group(3)
            item_number = match.group(4).upper()
            title = self._clean_header_title(match.group(5))
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized_text)
            content = normalized_text[start:end].strip()
            part = self._normalize_part(part_token or roman_part)

            if not self._section_is_relevant(
                form_type=form_type,
                item_number=item_number,
                part=part,
            ):
                continue
            if len(content) < 80:
                continue

            section_id = self._section_id(form_type=form_type, item_number=item_number, part=part)
            sections.append(
                FilingSection(
                    section_id=section_id,
                    item_label=f"Item {item_number}",
                    title=title or section_id.replace("_", " "),
                    part=part,
                    content=content,
                    start_offset=start,
                    end_offset=end,
                )
            )

        return sections

    def _download_filings(self, ticker: str, form_type: str, limit: int) -> None:
        try:
            downloader = Downloader(
                self.settings.sec_company_name,
                self.settings.sec_email_address,
                str(self.settings.sec_raw_dir),
            )
            logger.info(
                "downloading sec filings",
                extra={"ticker": ticker, "form_type": form_type, "limit": limit},
            )
            downloader.get(form_type, ticker, limit=limit, download_details=True)
        except HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                raise RuntimeError(
                    "SEC request was rejected with HTTP 403. "
                    "Set SEC_COMPANY_NAME and SEC_EMAIL_ADDRESS in .env to a real "
                    "organization name and contact email, then retry."
                ) from exc
            raise

    def _discover_filing_documents(self, ticker: str, form_type: str, limit: int) -> list[Path]:
        base_dir = self.settings.sec_raw_dir / "sec-edgar-filings" / ticker / form_type
        if not base_dir.exists():
            return []

        candidates = sorted(
            [
                path
                for path in base_dir.rglob("*")
                if path.is_file()
                and path.name.lower()
                in {"full-submission.txt", "primary-document.html", "primary_doc.html"}
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        unique_parents: set[Path] = set()
        selected: list[Path] = []
        for candidate in candidates:
            parent = candidate.parent
            if parent in unique_parents:
                continue
            unique_parents.add(parent)
            selected.append(candidate)
            if len(selected) >= limit:
                break
        return selected

    def _build_document_record(self, path: Path, ticker: str, form_type: str) -> DocumentRecord:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        primary_text = self._extract_primary_document(raw_text=raw_text, form_type=form_type)
        cleaned_text = self._clean_filing_text(primary_text)
        sections = self.extract_sections(cleaned_text, form_type=form_type)
        accession_number = self._extract_accession_number(path, raw_text)
        filed_at = self._extract_filed_at(raw_text)
        document_suffix = accession_number or path.parent.name.lower()
        document_id = f"{ticker.lower()}-{form_type.lower()}-{document_suffix}"

        return DocumentRecord(
            document_id=document_id,
            source_type="filing",
            ticker=ticker,
            title=f"{ticker} {form_type} filing",
            provider="sec-edgar-downloader",
            form_type=form_type,
            published_at=filed_at,
            as_of_date=filed_at.date() if filed_at else None,
            accession_number=accession_number,
            cik=self._extract_cik(raw_text),
            metadata_version=self.settings.sec_metadata_version,
            raw_checksum=self._checksum(raw_text),
            raw_path=str(path),
            cleaned_text=cleaned_text,
            sections=sections,
        )

    def _chunk_document(self, document: DocumentRecord) -> list[EvidenceChunk]:
        sections = document.sections or [
            FilingSection(
                section_id="full_document",
                item_label=document.form_type or "filing",
                title=document.title,
                content=document.cleaned_text,
                start_offset=0,
                end_offset=len(document.cleaned_text),
            )
        ]

        chunks: list[EvidenceChunk] = []
        for section in sections:
            section_chunks = self._splitter.split_text(section.content)
            for idx, chunk_text in enumerate(section_chunks):
                chunk_id = f"{document.document_id}-{section.section_id}-{idx:03d}"
                chunks.append(
                    EvidenceChunk(
                        chunk_id=chunk_id,
                        source_id=chunk_id,
                        document_id=document.document_id,
                        ticker=document.ticker,
                        title=f"{document.title} | {section.title}",
                        content=chunk_text,
                        document_type="filing",
                        provider=document.provider,
                        form_type=document.form_type,
                        section=section.section_id,
                        source_url=document.source_url,
                        published_at=document.published_at,
                        accession_number=document.accession_number,
                        chunk_index=idx,
                        metadata_version=document.metadata_version,
                    )
                )
        return chunks

    def _persist_document(self, document: DocumentRecord) -> str:
        form_type = cast(str, document.form_type)
        base_dir = self.settings.normalized_data_dir / "sec" / document.ticker / form_type
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{document.document_id}.json"
        output_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return str(output_path)

    def _persist_chunks(self, document: DocumentRecord, chunks: list[EvidenceChunk]) -> str:
        form_type = cast(str, document.form_type)
        base_dir = self.settings.chunked_data_dir / "sec" / document.ticker / form_type
        base_dir.mkdir(parents=True, exist_ok=True)
        output_path = base_dir / f"{document.document_id}.jsonl"
        payload = "\n".join(chunk.model_dump_json() for chunk in chunks)
        output_path.write_text(payload, encoding="utf-8")
        return str(output_path)

    def _extract_primary_document(self, raw_text: str, form_type: str) -> str:
        document_blocks = re.findall(
            r"<DOCUMENT>(.*?)</DOCUMENT>",
            raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not document_blocks:
            return raw_text

        for block in document_blocks:
            type_match = re.search(r"<TYPE>\s*([^\n<]+)", block, flags=re.IGNORECASE)
            text_match = re.search(r"<TEXT>(.*)", block, flags=re.IGNORECASE | re.DOTALL)
            if not type_match or not text_match:
                continue
            if type_match.group(1).strip().upper() == form_type.upper():
                return text_match.group(1)
        return raw_text

    def _clean_filing_text(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        cleaned = soup.get_text("\n")
        return self._normalize_text(cleaned)

    def _section_is_relevant(self, form_type: str, item_number: str, part: str | None) -> bool:
        if form_type == "8-K":
            return bool(item_number)
        allowed = TARGET_SECTIONS.get(form_type, ())
        if item_number not in allowed:
            return False
        if form_type == "10-Q":
            return (item_number == "2" and part == "part_i") or (
                item_number == "1A" and part == "part_ii"
            )
        return True

    def _section_id(self, form_type: str, item_number: str, part: str | None) -> str:
        return SECTION_KEYS.get(
            (form_type, item_number, part),
            f"{part + '_' if part else ''}item_{item_number.lower()}",
        )

    def _extract_accession_number(self, path: Path, raw_text: str) -> str | None:
        path_match = ACCESSION_RE.search(str(path))
        if path_match:
            return path_match.group(0)
        raw_match = ACCESSION_RE.search(raw_text)
        return raw_match.group(0) if raw_match else None

    def _extract_filed_at(self, raw_text: str) -> datetime | None:
        match = FILED_AS_OF_RE.search(raw_text)
        if not match:
            return None
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=UTC)
        except ValueError:
            return None

    def _extract_cik(self, raw_text: str) -> str | None:
        match = CIK_RE.search(raw_text)
        return match.group(1) if match else None

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _clean_header_title(self, title: str) -> str:
        title = title.replace("Table of Contents", "").strip(" .:-")
        return re.sub(r"\s+", " ", title)

    def _normalize_part(self, value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.lower().replace(" ", "_")
        if normalized in {"part_i", "part_ii", "part_iii", "part_iv"}:
            return normalized
        return None

    def _checksum(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _validate_sec_identity(self) -> None:
        company_name = self.settings.sec_company_name.strip()
        email_address = self.settings.sec_email_address.strip().lower()
        invalid_emails = {"you@example.com", "example@example.com", "test@example.com"}

        if not company_name or not email_address:
            raise ValueError("SEC_COMPANY_NAME and SEC_EMAIL_ADDRESS must be configured.")
        if email_address in invalid_emails or email_address.endswith("@example.com"):
            raise ValueError(
                "SEC_EMAIL_ADDRESS must be a real contact email accepted by SEC EDGAR."
            )

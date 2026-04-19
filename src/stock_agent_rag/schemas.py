from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class EvidenceRecord(BaseModel):
    source_id: str
    ticker: str
    title: str
    content: str
    document_type: Literal[
        "filing", "transcript", "news", "fundamentals", "note", "unknown"
    ] = "unknown"
    source_url: str | None = None
    published_at: datetime | None = None
    score: float = 0.0
    provider: str | None = None
    section: str | None = None
    form_type: str | None = None
    document_id: str | None = None
    accession_number: str | None = None
    chunk_index: int | None = None
    metadata_version: str | None = None
    speaker: str | None = None
    speaker_role: str | None = None
    publisher: str | None = None
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    ticker_relevance_score: float | None = None
    entity_title_match: bool | None = None
    entity_body_match: bool | None = None
    news_relevance_score: float | None = None
    news_relevance_tier: str | None = None
    source_quality_tier: str | None = None


class FundamentalsSnapshot(BaseModel):
    ticker: str
    as_of: datetime | None = None
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    source: str = "yfinance"


class AnalystFinding(BaseModel):
    finding: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_data: list[str] = Field(default_factory=list)
    finding_type: str | None = None


class AnalystOutput(BaseModel):
    summary: str
    findings: list[AnalystFinding] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ThesisFinding(BaseModel):
    analyst: str
    finding: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_data: list[str] = Field(default_factory=list)
    finding_type: str | None = None


class ThesisSectionInput(BaseModel):
    section_id: str
    title: str
    objective: str
    findings: list[ThesisFinding] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ThesisPreparation(BaseModel):
    sections: list[ThesisSectionInput] = Field(default_factory=list)


class ContradictionRecord(BaseModel):
    topic: str
    claim_a: str
    claim_b: str
    analyst_a: str
    analyst_b: str
    evidence_ids_a: list[str] = Field(default_factory=list)
    evidence_ids_b: list[str] = Field(default_factory=list)
    contradiction_kind: Literal[
        "direct_conflict",
        "time_horizon_tension",
        "evidence_quality_gap",
        "not_a_contradiction",
    ] = "direct_conflict"
    severity: Literal["low", "medium", "high"] = "medium"
    resolution_status: Literal["open", "explained", "resolved"] = "open"
    rationale: str | None = None


def replace_value(_: object, new: object) -> object:
    return new


class ResearchState(TypedDict, total=False):
    ticker: str
    question: str
    plan: Annotated[str, replace_value]
    fundamentals: Annotated[FundamentalsSnapshot, replace_value]
    retrieved_evidence: Annotated[list[EvidenceRecord], replace_value]
    fundamentals_evidence: Annotated[list[EvidenceRecord], replace_value]
    sentiment_evidence: Annotated[list[EvidenceRecord], replace_value]
    risk_evidence: Annotated[list[EvidenceRecord], replace_value]
    fundamentals_analysis: Annotated[AnalystOutput, replace_value]
    sentiment_analysis: Annotated[AnalystOutput, replace_value]
    risk_analysis: Annotated[AnalystOutput, replace_value]
    contradictions: Annotated[list[ContradictionRecord], replace_value]
    contradiction_summary: Annotated[str, replace_value]
    thesis_preparation: Annotated[ThesisPreparation, replace_value]
    node_metrics: Annotated[dict[str, dict[str, object]], replace_value]
    fundamentals_notes: Annotated[str, replace_value]
    sentiment_notes: Annotated[str, replace_value]
    risk_notes: Annotated[str, replace_value]
    report: Annotated[str, replace_value]
    initial_report: Annotated[str, replace_value]
    verification_status: Annotated[str, replace_value]
    verification_metrics: Annotated[dict[str, object], replace_value]
    verification_summary: Annotated[str, replace_value]
    repair_attempted: Annotated[bool, replace_value]
    repair_reason: Annotated[str, replace_value]
    repair_summary: Annotated[str, replace_value]


class ResearchRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    question: str = Field(
        default="Generate an evidence-backed investment thesis.",
        min_length=5,
    )


class ResearchResponse(BaseModel):
    ticker: str
    question: str
    plan: str
    report: str
    verification_status: str = "unknown"
    verification_summary: str
    retrieved_sources: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    model_metadata: dict[str, object] = Field(default_factory=dict)
    runtime_metrics: dict[str, int] = Field(default_factory=dict)
    retrieval_metrics: dict[str, object] = Field(default_factory=dict)
    estimated_cost_usd: float | None = None
    thesis_id: str | None = None
    thesis_storage_provider: str | None = None
    thesis_bucket: str | None = None
    thesis_object_key: str | None = None
    thesis_markdown_path: str | None = None
    latency_ms: float


class StoredObject(BaseModel):
    storage_provider: str
    bucket: str
    object_key: str
    content_type: str
    etag: str | None = None
    markdown_path: str | None = None


class ThesisArtifactSummary(BaseModel):
    thesis_id: str
    run_id: str
    ticker: str
    storage_provider: str
    bucket: str
    object_key: str
    markdown_path: str | None = None
    markdown_checksum: str
    thesis_hash: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    environment: str


class FilingSection(BaseModel):
    section_id: str
    item_label: str
    title: str
    part: str | None = None
    content: str
    start_offset: int
    end_offset: int


class TranscriptTurn(BaseModel):
    turn_id: str
    speaker: str
    speaker_role: str | None = None
    content: str
    order: int


class DocumentRecord(BaseModel):
    document_id: str
    source_type: Literal["filing", "transcript", "news", "analyst_note"]
    ticker: str
    title: str
    provider: str
    form_type: str | None = None
    published_at: datetime | None = None
    as_of_date: date | None = None
    source_url: str | None = None
    accession_number: str | None = None
    cik: str | None = None
    metadata_version: str = "1.0"
    raw_checksum: str
    raw_path: str
    cleaned_text: str
    publisher: str | None = None
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    ticker_relevance_score: float | None = None
    entity_title_match: bool | None = None
    entity_body_match: bool | None = None
    news_relevance_score: float | None = None
    news_relevance_tier: str | None = None
    source_quality_tier: str | None = None
    sections: list[FilingSection] = Field(default_factory=list)
    transcript_turns: list[TranscriptTurn] = Field(default_factory=list)


class EvidenceChunk(BaseModel):
    chunk_id: str
    source_id: str
    document_id: str
    ticker: str
    title: str
    content: str
    document_type: Literal[
        "filing", "transcript", "news", "fundamentals", "note", "unknown"
    ] = "filing"
    provider: str
    form_type: str | None = None
    section: str | None = None
    source_url: str | None = None
    published_at: datetime | None = None
    accession_number: str | None = None
    chunk_index: int
    metadata_version: str = "1.0"
    score: float = 0.0
    speaker: str | None = None
    speaker_role: str | None = None
    publisher: str | None = None
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    ticker_relevance_score: float | None = None
    entity_title_match: bool | None = None
    entity_body_match: bool | None = None
    news_relevance_score: float | None = None
    news_relevance_tier: str | None = None
    source_quality_tier: str | None = None

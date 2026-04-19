from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "evaluation" / "golden_set.json"

RELEASE_GATE_THRESHOLDS: dict[str, float] = {
    "citation_format_compliance": 1.0,
    "unsupported_numeric_claim_rate": 0.0,
    "off_ticker_evidence_rate": 0.02,
    "contradiction_surfacing_rate": 0.8,
    "pass_rate_after_repair": 0.8,
}


class GoldenSetCase(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    question: str = Field(min_length=5)
    sector: str = Field(min_length=2)
    market_regime: str = Field(min_length=2)
    expected_document_types: list[str] = Field(default_factory=list)
    relevant_source_ids: list[str] = Field(default_factory=list)
    required_issues: list[str] = Field(default_factory=list)
    prohibited_claims: list[str] = Field(default_factory=list)
    verdict_band: str = Field(min_length=2)
    requires_contradiction_review: bool = False


class GoldenSetManifest(BaseModel):
    version: str
    description: str
    cases: list[GoldenSetCase] = Field(default_factory=list)


def load_golden_set(path: Path | str | None = None) -> GoldenSetManifest:
    golden_set_path = Path(path) if path is not None else DEFAULT_GOLDEN_SET_PATH
    payload = json.loads(golden_set_path.read_text(encoding="utf-8"))
    return GoldenSetManifest.model_validate(payload)


def summarize_golden_set(manifest: GoldenSetManifest) -> dict[str, object]:
    sectors = sorted({case.sector for case in manifest.cases})
    tickers = sorted({case.ticker.upper() for case in manifest.cases})
    contradiction_cases = sum(1 for case in manifest.cases if case.requires_contradiction_review)
    return {
        "version": manifest.version,
        "case_count": len(manifest.cases),
        "sector_count": len(sectors),
        "sectors": sectors,
        "ticker_count": len(tickers),
        "contradiction_cases": contradiction_cases,
    }


def evaluate_release_gates(
    *,
    results: list[dict[str, object]],
    manifest: GoldenSetManifest,
    retrieval_k: int = 5,
) -> dict[str, object]:
    case_lookup = {
        _case_key(case.ticker, case.question): case
        for case in manifest.cases
    }
    matched_runs: list[tuple[GoldenSetCase, dict[str, object]]] = []
    unmatched_runs = 0
    for result in results:
        ticker = str(result.get("ticker") or "").upper()
        question = str(result.get("question") or "")
        case = case_lookup.get(_case_key(ticker, question))
        if case is None:
            unmatched_runs += 1
            continue
        matched_runs.append((case, result))

    total_runs = len(matched_runs)
    citation_compliant_runs = 0
    unsupported_numeric_claims = 0
    off_ticker_evidence_count = 0
    total_retrieved_evidence = 0
    contradiction_expected = 0
    contradiction_surfaced = 0
    repair_attempts = 0
    repair_passes = 0
    verification_passes = 0
    retrieval_labeled_cases = 0
    precision_sum = 0.0
    recall_sum = 0.0

    for case, result in matched_runs:
        verification_metrics = _coerce_dict(result.get("verification_metrics"))
        retrieval_metrics = _coerce_dict(result.get("retrieval_metrics"))

        malformed_citation_count = int(verification_metrics.get("malformed_citation_count", 0) or 0)
        prohibited_placeholder_count = int(
            verification_metrics.get("prohibited_placeholder_count", 0) or 0
        )
        uncited_numeric_claim_count = int(
            verification_metrics.get("uncited_numeric_claim_count", 0) or 0
        )
        if malformed_citation_count == 0 and prohibited_placeholder_count == 0:
            citation_compliant_runs += 1
        unsupported_numeric_claims += uncited_numeric_claim_count

        retrieved_count = int(retrieval_metrics.get("merged_retrieved_count", 0) or 0)
        total_retrieved_evidence += retrieved_count
        off_ticker_evidence_count += int(retrieval_metrics.get("off_ticker_evidence_count", 0) or 0)

        if case.relevant_source_ids:
            retrieval_labeled_cases += 1
            retrieved_source_ids = _extract_retrieved_source_ids(result)
            precision_sum += precision_at_k(
                retrieved_source_ids=retrieved_source_ids,
                relevant_source_ids=case.relevant_source_ids,
                k=retrieval_k,
            )
            recall_sum += recall_at_k(
                retrieved_source_ids=retrieved_source_ids,
                relevant_source_ids=case.relevant_source_ids,
                k=retrieval_k,
            )

        contradictions = result.get("contradictions")
        contradiction_count = len(contradictions) if isinstance(contradictions, list) else 0
        if case.requires_contradiction_review:
            contradiction_expected += 1
            if contradiction_count > 0:
                contradiction_surfaced += 1

        repair_attempted = bool(
            result.get("repair_attempted", verification_metrics.get("repair_attempted", False))
        )
        verification_status = str(result.get("verification_status") or "unknown").lower()
        if verification_status == "pass":
            verification_passes += 1
        if repair_attempted:
            repair_attempts += 1
            if verification_status == "pass":
                repair_passes += 1

    metrics = {
        "evaluated_case_count": total_runs,
        "golden_set_case_count": len(manifest.cases),
        "golden_set_coverage": round(_safe_ratio(total_runs, len(manifest.cases)), 4),
        "unmatched_run_count": unmatched_runs,
        "verification_pass_rate": round(_safe_ratio(verification_passes, total_runs), 4),
        "citation_format_compliance": round(_safe_ratio(citation_compliant_runs, total_runs), 4),
        "unsupported_numeric_claim_count": unsupported_numeric_claims,
        "unsupported_numeric_claim_rate": round(
            _safe_ratio(unsupported_numeric_claims, total_runs),
            4,
        ),
        "off_ticker_evidence_count": off_ticker_evidence_count,
        "off_ticker_evidence_rate": round(
            _safe_ratio(off_ticker_evidence_count, total_retrieved_evidence),
            4,
        ),
        "retrieval_labeled_case_count": retrieval_labeled_cases,
        "retrieval_label_coverage": round(
            _safe_ratio(retrieval_labeled_cases, total_runs),
            4,
        ),
        f"precision@{retrieval_k}": round(
            _safe_ratio_sum(precision_sum, retrieval_labeled_cases),
            4,
        ),
        f"recall@{retrieval_k}": round(_safe_ratio_sum(recall_sum, retrieval_labeled_cases), 4),
        "expected_contradiction_cases": contradiction_expected,
        "surfaced_contradiction_cases": contradiction_surfaced,
        "contradiction_surfacing_rate": round(
            _safe_ratio(contradiction_surfaced, contradiction_expected),
            4,
        ),
        "repair_attempt_count": repair_attempts,
        "pass_rate_after_repair": round(_safe_ratio(repair_passes, repair_attempts), 4),
    }
    gates = {
        "minimum_golden_set_coverage": metrics["golden_set_coverage"] >= 1.0,
        "citation_format_compliance": (
            float(metrics["citation_format_compliance"])
            >= RELEASE_GATE_THRESHOLDS["citation_format_compliance"]
        ),
        "unsupported_numeric_claim_rate": (
            float(metrics["unsupported_numeric_claim_rate"])
            <= RELEASE_GATE_THRESHOLDS["unsupported_numeric_claim_rate"]
        ),
        "off_ticker_evidence_rate": (
            float(metrics["off_ticker_evidence_rate"])
            <= RELEASE_GATE_THRESHOLDS["off_ticker_evidence_rate"]
        ),
        "contradiction_surfacing_rate": (
            float(metrics["contradiction_surfacing_rate"])
            >= RELEASE_GATE_THRESHOLDS["contradiction_surfacing_rate"]
        ),
        "pass_rate_after_repair": (
            repair_attempts == 0
            or float(metrics["pass_rate_after_repair"])
            >= RELEASE_GATE_THRESHOLDS["pass_rate_after_repair"]
        ),
    }
    return {
        "golden_set_summary": summarize_golden_set(manifest),
        "metrics": metrics,
        "thresholds": RELEASE_GATE_THRESHOLDS,
        "gates": gates,
        "status": "pass" if all(gates.values()) else "fail",
    }


def precision_at_k(
    *,
    retrieved_source_ids: list[str],
    relevant_source_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_source_ids[:k]
    if not top_k:
        return 0.0
    relevant = {source_id.strip() for source_id in relevant_source_ids if source_id.strip()}
    hits = sum(1 for source_id in top_k if source_id in relevant)
    return hits / len(top_k)


def recall_at_k(
    *,
    retrieved_source_ids: list[str],
    relevant_source_ids: list[str],
    k: int,
) -> float:
    relevant = {source_id.strip() for source_id in relevant_source_ids if source_id.strip()}
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved_source_ids[:k]
    hits = sum(1 for source_id in top_k if source_id in relevant)
    return hits / len(relevant)


def _case_key(ticker: str, question: str) -> tuple[str, str]:
    return (ticker.strip().upper(), question.strip())


def _coerce_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _safe_ratio_sum(total: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return total / denominator


def _extract_retrieved_source_ids(result: dict[str, object]) -> list[str]:
    retrieved_sources = result.get("retrieved_sources")
    if isinstance(retrieved_sources, list):
        normalized = [str(item).strip() for item in retrieved_sources if str(item).strip()]
        if normalized:
            return normalized

    retrieved_evidence = result.get("retrieved_evidence")
    if isinstance(retrieved_evidence, list):
        normalized: list[str] = []
        for item in retrieved_evidence:
            source_id = getattr(item, "source_id", None)
            if source_id is None and isinstance(item, dict):
                source_id = item.get("source_id")
            if source_id is not None:
                normalized.append(str(source_id).strip())
        if normalized:
            return normalized

    return []

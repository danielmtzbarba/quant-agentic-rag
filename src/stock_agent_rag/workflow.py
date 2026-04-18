from __future__ import annotations

import re
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .config import get_settings
from .prompts import (
    CONTRADICTION_REVIEW_PROMPT,
    FUNDAMENTALS_ANALYST_PROMPT,
    PLANNER_PROMPT,
    RISK_ANALYST_PROMPT,
    SENTIMENT_ANALYST_PROMPT,
    THESIS_PROMPT,
    VERIFIER_PROMPT,
)
from .schemas import (
    AnalystOutput,
    ContradictionRecord,
    EvidenceRecord,
    FundamentalsSnapshot,
    ResearchState,
    ThesisFinding,
    ThesisPreparation,
    ThesisSectionInput,
)
from .telemetry import (
    estimate_cost_usd,
    extract_retry_count,
    extract_timeout_count,
)
from .tools import (
    build_evidence_context,
    fetch_fundamentals_snapshot,
    local_corpus_search,
    merge_evidence_sets,
)

SOURCE_CITATION_RE = re.compile(r"\[source:([^\]]+)\]")
TOKEN_RE = re.compile(r"[a-z0-9]+")
POSITIVE_FINDING_TOKENS = {
    "strong",
    "improved",
    "improving",
    "accelerating",
    "constructive",
    "upside",
    "healthy",
    "resilient",
    "favorable",
    "raised",
    "positive",
    "growth",
}
NEGATIVE_FINDING_TOKENS = {
    "weak",
    "weaker",
    "declining",
    "deteriorating",
    "pressure",
    "headwind",
    "risk",
    "bear",
    "downside",
    "concern",
    "uncertain",
    "negative",
    "constrained",
    "overvalued",
}
SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1}
POLARITY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "remain",
    "remains",
    "still",
    "more",
    "less",
    "very",
    "could",
    "would",
    "should",
    "management",
    "company",
    "business",
    "market",
}
NORMALIZED_CONTRADICTION_TOPICS = {
    "guidance": "guidance_quality",
    "outlook": "guidance_quality",
    "tone": "management_tone",
    "sentiment": "management_tone",
    "demand": "demand_outlook",
    "orders": "demand_outlook",
    "revenue": "revenue_growth",
    "growth": "revenue_growth",
    "margin": "gross_margin",
    "margins": "gross_margin",
    "cash": "cash_flow",
    "liquidity": "cash_flow",
    "balance": "balance_sheet",
    "debt": "balance_sheet",
    "valuation": "valuation",
    "multiple": "valuation",
    "execution": "execution_risk",
    "supply": "execution_risk",
    "regulation": "regulatory_legal",
    "regulatory": "regulatory_legal",
    "legal": "regulatory_legal",
}
TIME_HORIZON_TOKENS = {
    "quarter",
    "quarterly",
    "next",
    "later",
    "forward",
    "upcoming",
    "nearterm",
    "longterm",
    "year",
    "annual",
    "future",
}


class ContradictionReviewOutput(BaseModel):
    is_contradiction: bool
    contradiction_kind: str = Field(
        pattern="^(direct_conflict|time_horizon_tension|evidence_quality_gap|not_a_contradiction)$"
    )
    normalized_topic: str = Field(
        pattern=(
            "^(demand_outlook|revenue_growth|gross_margin|cash_flow|balance_sheet|"
            "valuation|execution_risk|regulatory_legal|management_tone|guidance_quality|other)$"
        )
    )
    severity: str = Field(pattern="^(low|medium|high)$")
    resolution_status: str = Field(pattern="^(open|explained|resolved)$")
    rationale: str
    supporting_evidence_ids_a: list[str] = Field(default_factory=list)
    supporting_evidence_ids_b: list[str] = Field(default_factory=list)


def _get_model() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(model=settings.model_name, temperature=0)


def _get_structured_model():
    return _get_model().with_structured_output(AnalystOutput, include_raw=True)


def _get_contradiction_review_model():
    return _get_model().with_structured_output(ContradictionReviewOutput, include_raw=True)


def _extract_node_metrics(
    response: object,
    *,
    model_name: str,
    temperature: float,
) -> dict[str, object]:
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = (
        response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}
    )

    input_tokens = (
        usage_metadata.get("input_tokens")
        or token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or 0
    )
    output_tokens = (
        usage_metadata.get("output_tokens")
        or token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or 0
    )
    total_tokens = (
        usage_metadata.get("total_tokens")
        or token_usage.get("total_tokens")
        or (int(input_tokens) + int(output_tokens))
    )
    resolved_model_name = (
        response_metadata.get("model_name") if isinstance(response_metadata, dict) else None
    ) or model_name
    retry_count = extract_retry_count(response_metadata)
    timeout_count = extract_timeout_count(response_metadata)
    estimated_cost = estimate_cost_usd(
        model_name=resolved_model_name,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
    )

    return {
        "model_name": resolved_model_name,
        "temperature": temperature,
        "provider": "openai",
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
        "retry_count": retry_count,
        "timeout_count": timeout_count,
        "estimated_cost_usd": estimated_cost,
    }


def _record_node_metrics(
    state: ResearchState,
    *,
    node_name: str,
    response: object,
    started_at: float,
    model_name: str,
    temperature: float,
) -> dict[str, dict[str, object]]:
    node_metrics = dict(state.get("node_metrics", {}))
    metrics = _extract_node_metrics(response, model_name=model_name, temperature=temperature)
    metrics["latency_ms"] = round((perf_counter() - started_at) * 1000, 2)
    node_metrics[node_name] = metrics
    return node_metrics


def planner_node(state: ResearchState) -> dict:
    model = _get_model()
    started_at = perf_counter()
    response = model.invoke(
        [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=f"Ticker: {state['ticker']}\nQuestion: {state['question']}"),
        ]
    )
    return {
        "plan": response.content,
        "node_metrics": _record_node_metrics(
            state,
            node_name="planner",
            response=response,
            started_at=started_at,
            model_name=model.model_name,
            temperature=float(model.temperature or 0),
        ),
    }


def fundamentals_retrieval_node(state: ResearchState) -> dict:
    try:
        snapshot = fetch_fundamentals_snapshot(state["ticker"])
    except Exception as exc:
        snapshot = FundamentalsSnapshot(
            ticker=state["ticker"],
            metrics={"error": str(exc)},
            source="yfinance",
        )
    return {"fundamentals": snapshot}


def fundamentals_corpus_retrieval_node(state: ResearchState) -> dict:
    query = f"{state['ticker']} {state['question']} filings fundamentals mda financial statements"
    evidence = local_corpus_search(query=query, ticker=state["ticker"], profile="fundamentals")
    return {"fundamentals_evidence": evidence}


def sentiment_corpus_retrieval_node(state: ResearchState) -> dict:
    query = f"{state['ticker']} {state['question']} transcripts news sentiment management guidance"
    evidence = local_corpus_search(query=query, ticker=state["ticker"], profile="sentiment")
    return {"sentiment_evidence": evidence}


def risk_corpus_retrieval_node(state: ResearchState) -> dict:
    query = (
        f"{state['ticker']} {state['question']} risk factors legal regulation "
        "balance sheet news"
    )
    evidence = local_corpus_search(query=query, ticker=state["ticker"], profile="risk")
    return {"risk_evidence": evidence}


def aggregate_evidence_node(state: ResearchState) -> dict:
    merged = merge_evidence_sets(
        state.get("fundamentals_evidence", []),
        state.get("sentiment_evidence", []),
        state.get("risk_evidence", []),
    )
    return {"retrieved_evidence": merged}

def _analyst_prompt(state: ResearchState, evidence_key: str) -> str:
    fundamentals = state.get("fundamentals")
    evidence = state.get(evidence_key, [])
    if isinstance(fundamentals, FundamentalsSnapshot):
        fundamentals_block = fundamentals.model_dump_json(indent=2)
    else:
        fundamentals_block = "No fundamentals."
    evidence_block = build_evidence_context(evidence)
    return (
        f"Plan:\n{state.get('plan', '')}\n\n"
        f"Fundamentals:\n{fundamentals_block}\n\n"
        f"Evidence:\n{evidence_block}"
    )


def fundamentals_analyst_node(state: ResearchState) -> dict:
    model = _get_structured_model()
    started_at = perf_counter()
    response = model.invoke(
        [
            SystemMessage(content=FUNDAMENTALS_ANALYST_PROMPT),
            HumanMessage(content=_analyst_prompt(state, "fundamentals_evidence")),
        ]
    )
    parsed: AnalystOutput = response["parsed"]
    raw = response["raw"]
    return {
        "fundamentals_analysis": parsed,
        "fundamentals_notes": parsed.summary,
        "node_metrics": _record_node_metrics(
            state,
            node_name="fundamentals_analyst",
            response=raw,
            started_at=started_at,
            model_name=get_settings().model_name,
            temperature=0.0,
        ),
    }


def sentiment_analyst_node(state: ResearchState) -> dict:
    model = _get_structured_model()
    started_at = perf_counter()
    response = model.invoke(
        [
            SystemMessage(content=SENTIMENT_ANALYST_PROMPT),
            HumanMessage(content=_analyst_prompt(state, "sentiment_evidence")),
        ]
    )
    parsed: AnalystOutput = response["parsed"]
    raw = response["raw"]
    return {
        "sentiment_analysis": parsed,
        "sentiment_notes": parsed.summary,
        "node_metrics": _record_node_metrics(
            state,
            node_name="sentiment_analyst",
            response=raw,
            started_at=started_at,
            model_name=get_settings().model_name,
            temperature=0.0,
        ),
    }


def risk_analyst_node(state: ResearchState) -> dict:
    model = _get_structured_model()
    started_at = perf_counter()
    response = model.invoke(
        [
            SystemMessage(content=RISK_ANALYST_PROMPT),
            HumanMessage(content=_analyst_prompt(state, "risk_evidence")),
        ]
    )
    parsed: AnalystOutput = response["parsed"]
    raw = response["raw"]
    return {
        "risk_analysis": parsed,
        "risk_notes": parsed.summary,
        "node_metrics": _record_node_metrics(
            state,
            node_name="risk_analyst",
            response=raw,
            started_at=started_at,
            model_name=get_settings().model_name,
            temperature=0.0,
        ),
    }


def _analysis_block(state: ResearchState, key: str) -> str:
    analysis = state.get(key)
    if isinstance(analysis, AnalystOutput):
        return analysis.model_dump_json(indent=2)
    return "No structured analysis."


def _iter_analyses(state: ResearchState) -> list[tuple[str, AnalystOutput]]:
    analyses: list[tuple[str, AnalystOutput]] = []
    for label, key in (
        ("fundamentals", "fundamentals_analysis"),
        ("sentiment", "sentiment_analysis"),
        ("risk", "risk_analysis"),
    ):
        analysis = state.get(key)
        if isinstance(analysis, AnalystOutput):
            analyses.append((label, analysis))
    return analyses


def _finding_to_thesis_finding(analyst_label: str, finding) -> ThesisFinding:
    return ThesisFinding(
        analyst=analyst_label,
        finding=finding.finding,
        evidence_ids=list(finding.evidence_ids),
        confidence=finding.confidence,
        missing_data=list(finding.missing_data),
        finding_type=finding.finding_type,
    )


def _normalize_tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in POLARITY_STOPWORDS and len(token) > 2
    }


def _finding_polarity(finding_text: str, finding_type: str | None) -> int:
    tokens = _normalize_tokens(finding_text)
    if finding_type:
        tokens.update(_normalize_tokens(finding_type))
    positive_hits = len(tokens & POSITIVE_FINDING_TOKENS)
    negative_hits = len(tokens & NEGATIVE_FINDING_TOKENS)
    if positive_hits > negative_hits:
        return 1
    if negative_hits > positive_hits:
        return -1
    return 0


def _topic_tokens(finding_text: str, finding_type: str | None) -> set[str]:
    tokens = _normalize_tokens(finding_text)
    if finding_type:
        tokens.update(_normalize_tokens(finding_type))
    return {
        token
        for token in tokens
        if token not in POSITIVE_FINDING_TOKENS and token not in NEGATIVE_FINDING_TOKENS
    }


def _topic_label(shared_tokens: set[str]) -> str:
    if not shared_tokens:
        return "general thesis"
    ordered = sorted(shared_tokens)
    return " / ".join(ordered[:3])


def _normalize_contradiction_topic(shared_tokens: set[str]) -> str:
    for token in sorted(shared_tokens):
        mapped = NORMALIZED_CONTRADICTION_TOPICS.get(token)
        if mapped:
            return mapped
    return "other"


def _contradiction_severity(shared_tokens: set[str], evidence_overlap: bool) -> str:
    if evidence_overlap and shared_tokens:
        return "high"
    if len(shared_tokens) >= 2:
        return "medium"
    return "low"


def _collect_evidence_lookup(state: ResearchState) -> dict[str, EvidenceRecord]:
    lookup: dict[str, EvidenceRecord] = {}
    for key in (
        "retrieved_evidence",
        "fundamentals_evidence",
        "sentiment_evidence",
        "risk_evidence",
    ):
        for record in state.get(key, []):
            if isinstance(record, EvidenceRecord):
                lookup[record.source_id] = record
    return lookup


def _format_evidence_snippets(evidence_ids: list[str], lookup: dict[str, EvidenceRecord]) -> str:
    if not evidence_ids:
        return "No evidence ids."
    snippets: list[str] = []
    for source_id in evidence_ids[:4]:
        record = lookup.get(source_id)
        if record is None:
            snippets.append(f"[source:{source_id}] missing from retrieval context")
            continue
        snippets.append(
            f"[source:{source_id}] {record.document_type} | {record.title}\n"
            f"published_at={record.published_at} section={record.section} "
            f"speaker_role={record.speaker_role} publisher={record.publisher}\n"
            f"{record.content[:400]}"
        )
    return "\n\n".join(snippets)


def _review_payload(record: ContradictionRecord, lookup: dict[str, EvidenceRecord]) -> str:
    return (
        f"Candidate topic: {record.topic}\n"
        f"Analyst A: {record.analyst_a}\n"
        f"Claim A: {record.claim_a}\n"
        f"Evidence A ids: {', '.join(record.evidence_ids_a) or 'none'}\n"
        f"Evidence A snippets:\n{_format_evidence_snippets(record.evidence_ids_a, lookup)}\n\n"
        f"Analyst B: {record.analyst_b}\n"
        f"Claim B: {record.claim_b}\n"
        f"Evidence B ids: {', '.join(record.evidence_ids_b) or 'none'}\n"
        f"Evidence B snippets:\n{_format_evidence_snippets(record.evidence_ids_b, lookup)}"
    )


def _fallback_review_contradiction(record: ContradictionRecord) -> ContradictionRecord:
    tokens_a = _normalize_tokens(record.claim_a)
    tokens_b = _normalize_tokens(record.claim_b)
    shared_tokens = tokens_a & tokens_b
    horizon_tension = bool((tokens_a | tokens_b) & TIME_HORIZON_TOKENS)
    evidence_overlap = bool(set(record.evidence_ids_a) & set(record.evidence_ids_b))

    contradiction_kind = "time_horizon_tension" if horizon_tension else "direct_conflict"
    resolution_status = "explained" if horizon_tension else "open"
    severity = _contradiction_severity(shared_tokens, evidence_overlap)
    rationale = (
        "Shortlisted heuristic conflict reviewed with deterministic fallback. "
        f"Shared topic tokens={', '.join(sorted(shared_tokens)[:4]) or 'none'}."
    )

    return ContradictionRecord(
        topic=_normalize_contradiction_topic(shared_tokens),
        claim_a=record.claim_a,
        claim_b=record.claim_b,
        analyst_a=record.analyst_a,
        analyst_b=record.analyst_b,
        evidence_ids_a=list(record.evidence_ids_a),
        evidence_ids_b=list(record.evidence_ids_b),
        contradiction_kind=contradiction_kind,
        severity=severity,
        resolution_status=resolution_status,
        rationale=rationale,
    )


def contradiction_check_node(state: ResearchState) -> dict:
    contradictions: list[ContradictionRecord] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    analyses = _iter_analyses(state)

    for idx, (analyst_a, analysis_a) in enumerate(analyses):
        for analyst_b, analysis_b in analyses[idx + 1 :]:
            for finding_a in analysis_a.findings:
                polarity_a = _finding_polarity(finding_a.finding, finding_a.finding_type)
                if polarity_a == 0:
                    continue
                topic_a = _topic_tokens(finding_a.finding, finding_a.finding_type)
                if not topic_a:
                    continue

                for finding_b in analysis_b.findings:
                    polarity_b = _finding_polarity(finding_b.finding, finding_b.finding_type)
                    if polarity_b == 0 or polarity_a == polarity_b:
                        continue

                    topic_b = _topic_tokens(finding_b.finding, finding_b.finding_type)
                    shared_tokens = topic_a & topic_b
                    if not shared_tokens:
                        continue

                    evidence_overlap = bool(
                        set(finding_a.evidence_ids) & set(finding_b.evidence_ids)
                    )
                    key = (
                        analyst_a,
                        analyst_b,
                        finding_a.finding.strip().lower(),
                        finding_b.finding.strip().lower(),
                    )
                    reverse_key = (
                        analyst_b,
                        analyst_a,
                        finding_b.finding.strip().lower(),
                        finding_a.finding.strip().lower(),
                    )
                    if key in seen_keys or reverse_key in seen_keys:
                        continue

                    seen_keys.add(key)
                    contradictions.append(
                        ContradictionRecord(
                            topic=_topic_label(shared_tokens),
                            claim_a=finding_a.finding,
                            claim_b=finding_b.finding,
                            analyst_a=analyst_a,
                            analyst_b=analyst_b,
                            evidence_ids_a=list(finding_a.evidence_ids),
                            evidence_ids_b=list(finding_b.evidence_ids),
                            contradiction_kind="direct_conflict",
                            severity=_contradiction_severity(shared_tokens, evidence_overlap),
                            resolution_status="open",
                            rationale=(
                                "Heuristic shortlist candidate from "
                                "opposite-polarity analyst findings."
                            ),
                        )
                    )

    contradictions.sort(
        key=lambda item: (SEVERITY_ORDER.get(item.severity, 0), item.topic),
        reverse=True,
    )
    if contradictions:
        preview = "; ".join(
            f"{item.analyst_a} vs {item.analyst_b} on {item.topic} ({item.severity})"
            for item in contradictions[:5]
        )
        summary = (
            f"Detected {len(contradictions)} cross-analyst contradictions. "
            f"Preview: {preview}."
        )
    else:
        summary = "No cross-analyst contradictions detected."

    return {
        "contradictions": contradictions,
        "contradiction_summary": summary,
    }


def contradiction_review_node(state: ResearchState) -> dict:
    candidates = state.get("contradictions", [])
    if not candidates:
        return {
            "contradictions": [],
            "contradiction_summary": "No cross-analyst contradictions detected.",
        }

    lookup = _collect_evidence_lookup(state)
    reviewed: list[ContradictionRecord] = []
    model = None
    started_at = perf_counter()
    raw_response: object | None = None

    try:
        model = _get_contradiction_review_model()
        for candidate in candidates:
            response = model.invoke(
                [
                    SystemMessage(content=CONTRADICTION_REVIEW_PROMPT),
                    HumanMessage(content=_review_payload(candidate, lookup)),
                ]
            )
            parsed: ContradictionReviewOutput = response["parsed"]
            raw_response = response["raw"]
            if not parsed.is_contradiction or parsed.contradiction_kind == "not_a_contradiction":
                continue
            reviewed.append(
                ContradictionRecord(
                    topic=parsed.normalized_topic,
                    claim_a=candidate.claim_a,
                    claim_b=candidate.claim_b,
                    analyst_a=candidate.analyst_a,
                    analyst_b=candidate.analyst_b,
                    evidence_ids_a=(
                        parsed.supporting_evidence_ids_a or list(candidate.evidence_ids_a)
                    ),
                    evidence_ids_b=(
                        parsed.supporting_evidence_ids_b or list(candidate.evidence_ids_b)
                    ),
                    contradiction_kind=parsed.contradiction_kind,
                    severity=parsed.severity,
                    resolution_status=parsed.resolution_status,
                    rationale=parsed.rationale,
                )
            )
    except Exception:
        reviewed = [_fallback_review_contradiction(candidate) for candidate in candidates]

    reviewed.sort(
        key=lambda item: (SEVERITY_ORDER.get(item.severity, 0), item.topic),
        reverse=True,
    )
    if reviewed:
        preview = "; ".join(
            (
                f"{item.analyst_a} vs {item.analyst_b} on {item.topic} "
                f"[{item.contradiction_kind}, {item.severity}, {item.resolution_status}]"
            )
            for item in reviewed[:5]
        )
        summary = f"Reviewed {len(reviewed)} material contradictions. Preview: {preview}."
    else:
        summary = "Reviewed contradiction candidates and found no material contradictions."

    result = {
        "contradictions": reviewed,
        "contradiction_summary": summary,
    }
    if model is not None and raw_response is not None:
        result["node_metrics"] = _record_node_metrics(
            state,
            node_name="contradiction_review",
            response=raw_response,
            started_at=started_at,
            model_name=get_settings().model_name,
            temperature=0.0,
        )
    return result


def _bucket_for_finding(analyst_label: str, thesis_finding: ThesisFinding) -> str:
    finding_type = (thesis_finding.finding_type or "").lower()
    finding_text = thesis_finding.finding.lower()

    if analyst_label == "risk":
        return "key_risks"

    negative_tokens = {
        "risk",
        "weakness",
        "constraint",
        "headwind",
        "bear",
        "downside",
        "pressure",
        "uncertain",
        "concern",
        "hype",
        "overvalued",
    }
    if finding_type in negative_tokens or any(token in finding_text for token in negative_tokens):
        return "bear_case"
    return "bull_case"


def _dedupe_evidence_ids(findings: list[ThesisFinding]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for finding in findings:
        for source_id in finding.evidence_ids:
            cleaned = source_id.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                ordered.append(cleaned)
    return ordered


def _top_findings_for_summary(prepared_findings: list[ThesisFinding]) -> list[ThesisFinding]:
    ranked = sorted(prepared_findings, key=lambda item: item.confidence, reverse=True)
    return ranked[:4]


def thesis_preparation_node(state: ResearchState) -> dict:
    buckets: dict[str, list[ThesisFinding]] = {
        "executive_summary": [],
        "bull_case": [],
        "bear_case": [],
        "key_risks": [],
        "evidence_gaps": [],
    }

    all_findings: list[ThesisFinding] = []
    for analyst_label, analysis in _iter_analyses(state):
        for finding in analysis.findings:
            thesis_finding = _finding_to_thesis_finding(analyst_label, finding)
            all_findings.append(thesis_finding)
            buckets[_bucket_for_finding(analyst_label, thesis_finding)].append(thesis_finding)

            if thesis_finding.missing_data:
                buckets["evidence_gaps"].append(thesis_finding)

        if analysis.evidence_gaps:
            for gap in analysis.evidence_gaps:
                buckets["evidence_gaps"].append(
                    ThesisFinding(
                        analyst=analyst_label,
                        finding=gap,
                        evidence_ids=[],
                        confidence=analysis.overall_confidence or 0.3,
                        missing_data=[gap],
                        finding_type="evidence_gap",
                    )
                )

    buckets["executive_summary"] = _top_findings_for_summary(all_findings)

    sections = [
        ThesisSectionInput(
            section_id="executive_summary",
            title="Executive Summary",
            objective="Summarize the highest-signal conclusions across analysts.",
            findings=buckets["executive_summary"],
            evidence_ids=_dedupe_evidence_ids(buckets["executive_summary"]),
        ),
        ThesisSectionInput(
            section_id="bull_case",
            title="Bull Case",
            objective="Present the strongest upside arguments supported by evidence.",
            findings=buckets["bull_case"],
            evidence_ids=_dedupe_evidence_ids(buckets["bull_case"]),
        ),
        ThesisSectionInput(
            section_id="bear_case",
            title="Bear Case",
            objective="Present the strongest downside arguments and counterpoints.",
            findings=buckets["bear_case"],
            evidence_ids=_dedupe_evidence_ids(buckets["bear_case"]),
        ),
        ThesisSectionInput(
            section_id="key_risks",
            title="Key Risks",
            objective="Highlight material risks that could impair the thesis.",
            findings=buckets["key_risks"],
            evidence_ids=_dedupe_evidence_ids(buckets["key_risks"]),
        ),
        ThesisSectionInput(
            section_id="evidence_gaps",
            title="Evidence Gaps",
            objective="Call out unresolved missing data and weakly supported areas.",
            findings=buckets["evidence_gaps"],
            evidence_ids=_dedupe_evidence_ids(buckets["evidence_gaps"]),
        ),
    ]

    preparation = ThesisPreparation(sections=sections)
    return {"thesis_preparation": preparation}


def _extract_cited_source_ids(report: str) -> set[str]:
    return {match.strip() for match in SOURCE_CITATION_RE.findall(report) if match.strip()}


def _structured_grounding_metrics(state: ResearchState, report: str) -> dict[str, object]:
    cited_source_ids = _extract_cited_source_ids(report)
    grounded_findings = 0
    partially_grounded_findings = 0
    unsupported_findings = 0
    findings_without_evidence_ids = 0
    unsupported_labels: list[str] = []

    for analyst_label, analysis in _iter_analyses(state):
        for idx, finding in enumerate(analysis.findings, start=1):
            expected_ids = [
                source_id.strip() for source_id in finding.evidence_ids if source_id.strip()
            ]
            if not expected_ids:
                findings_without_evidence_ids += 1
                unsupported_findings += 1
                unsupported_labels.append(f"{analyst_label}#{idx}")
                continue

            matched = [source_id for source_id in expected_ids if source_id in cited_source_ids]
            if len(matched) == len(expected_ids):
                grounded_findings += 1
            elif matched:
                partially_grounded_findings += 1
                unsupported_labels.append(f"{analyst_label}#{idx}")
            else:
                unsupported_findings += 1
                unsupported_labels.append(f"{analyst_label}#{idx}")

    return {
        "grounded_findings": grounded_findings,
        "partially_grounded_findings": partially_grounded_findings,
        "unsupported_findings": unsupported_findings,
        "findings_without_evidence_ids": findings_without_evidence_ids,
        "unsupported_labels": unsupported_labels,
    }


def _structured_grounding_summary(metrics: dict[str, object]) -> str:
    unsupported_labels = metrics.get("unsupported_labels", [])
    if isinstance(unsupported_labels, list):
        unsupported_preview = ", ".join(str(label) for label in unsupported_labels[:8]) or "none"
    else:
        unsupported_preview = "none"
    return (
        "Structured grounding: "
        f"grounded_findings={metrics.get('grounded_findings', 0)}, "
        f"partially_grounded_findings={metrics.get('partially_grounded_findings', 0)}, "
        f"unsupported_findings={metrics.get('unsupported_findings', 0)}, "
        f"findings_without_evidence_ids={metrics.get('findings_without_evidence_ids', 0)}, "
        f"unsupported_labels={unsupported_preview}."
    )


def thesis_node(state: ResearchState) -> dict:
    model = _get_model()
    started_at = perf_counter()
    evidence = state.get("retrieved_evidence", [])
    evidence_ids = ", ".join(record.source_id for record in evidence) if evidence else "none"
    thesis_preparation = state.get("thesis_preparation")
    thesis_input_block = (
        thesis_preparation.model_dump_json(indent=2)
        if isinstance(thesis_preparation, ThesisPreparation)
        else "No thesis preparation."
    )
    response = model.invoke(
        [
            SystemMessage(content=THESIS_PROMPT),
            HumanMessage(
                content=(
                    f"Ticker: {state['ticker']}\n"
                    f"Question: {state['question']}\n"
                    f"Plan:\n{state.get('plan', '')}\n\n"
                    f"Contradictions:\n{state.get('contradiction_summary', 'None.')}\n\n"
                    f"Prepared Thesis Sections:\n{thesis_input_block}\n\n"
                    f"Available source ids: {evidence_ids}"
                )
            ),
        ]
    )
    return {
        "report": response.content,
        "node_metrics": _record_node_metrics(
            state,
            node_name="thesis",
            response=response,
            started_at=started_at,
            model_name=model.model_name,
            temperature=float(model.temperature or 0),
        ),
    }


def verifier_node(state: ResearchState) -> dict:
    settings = get_settings()
    model = _get_model()
    started_at = perf_counter()
    evidence = state.get("retrieved_evidence", [])
    source_ids = [record.source_id for record in evidence]
    report = state.get("report", "")
    cited_source_ids = _extract_cited_source_ids(report)
    cited_count = sum(1 for source_id in source_ids if source_id in cited_source_ids)
    coverage = cited_count / len(source_ids) if source_ids else 0.0
    structured_findings = sum(
        len(analysis.findings)
        for _, analysis in _iter_analyses(state)
    )
    contradictions = state.get("contradictions", [])
    missing_data_count = sum(
        len(finding.missing_data)
        for _, analysis in _iter_analyses(state)
        for finding in analysis.findings
    )
    grounding_metrics = _structured_grounding_metrics(state, report)
    grounding_summary = _structured_grounding_summary(grounding_metrics)
    unsupported_findings = int(grounding_metrics.get("unsupported_findings", 0))
    partially_grounded_findings = int(grounding_metrics.get("partially_grounded_findings", 0))
    deterministic_status = (
        "fail"
        if unsupported_findings > settings.verifier_max_unsupported_findings
        or partially_grounded_findings > settings.verifier_max_partially_grounded_findings
        else "pass"
    )
    heuristic_summary = (
        f"Deterministic verifier status={deterministic_status.upper()}. "
        f"Retrieved {len(source_ids)} sources. "
        f"Structured findings={structured_findings}. "
        f"Contradictions={len(contradictions)}. "
        f"Missing-data flags={missing_data_count}. "
        f"Report cited {cited_count} retrieved source ids. "
        f"Heuristic citation coverage={coverage:.2%}. "
        f"{grounding_summary}"
    )
    response = model.invoke(
        [
            SystemMessage(content=VERIFIER_PROMPT),
            HumanMessage(
                content=(
                    f"{heuristic_summary}\n\n"
                    f"Fundamentals Analysis:\n{_analysis_block(state, 'fundamentals_analysis')}\n\n"
                    f"Sentiment Analysis:\n{_analysis_block(state, 'sentiment_analysis')}\n\n"
                    f"Risk Analysis:\n{_analysis_block(state, 'risk_analysis')}\n\n"
                    f"Contradiction Summary:\n{state.get('contradiction_summary', 'None.')}\n\n"
                    f"Report:\n{state.get('report', '')}"
                )
            ),
        ]
    )
    verification_metrics = {
        **grounding_metrics,
        "retrieved_sources": len(source_ids),
        "structured_findings": structured_findings,
        "contradictions": len(contradictions),
        "missing_data_flags": missing_data_count,
        "cited_retrieved_sources": cited_count,
        "citation_coverage": coverage,
        "deterministic_status": deterministic_status,
    }
    return {
        "verification_status": deterministic_status,
        "verification_metrics": verification_metrics,
        "verification_summary": f"{heuristic_summary}\n\n{response.content}",
        "node_metrics": _record_node_metrics(
            state,
            node_name="verifier",
            response=response,
            started_at=started_at,
            model_name=model.model_name,
            temperature=float(model.temperature or 0),
        ),
    }


def build_app():
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("fundamentals_retrieval", fundamentals_retrieval_node)
    workflow.add_node("fundamentals_corpus_retrieval", fundamentals_corpus_retrieval_node)
    workflow.add_node("sentiment_corpus_retrieval", sentiment_corpus_retrieval_node)
    workflow.add_node("risk_corpus_retrieval", risk_corpus_retrieval_node)
    workflow.add_node("aggregate_evidence", aggregate_evidence_node)
    workflow.add_node("fundamentals_analyst", fundamentals_analyst_node)
    workflow.add_node("sentiment_analyst", sentiment_analyst_node)
    workflow.add_node("risk_analyst", risk_analyst_node)
    workflow.add_node("contradiction_check", contradiction_check_node)
    workflow.add_node("contradiction_review", contradiction_review_node)
    workflow.add_node("thesis_preparation", thesis_preparation_node)
    workflow.add_node("thesis", thesis_node)
    workflow.add_node("verifier", verifier_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "fundamentals_retrieval")
    workflow.add_edge("planner", "fundamentals_corpus_retrieval")
    workflow.add_edge("planner", "sentiment_corpus_retrieval")
    workflow.add_edge("planner", "risk_corpus_retrieval")
    workflow.add_edge("fundamentals_retrieval", "fundamentals_analyst")
    workflow.add_edge("fundamentals_corpus_retrieval", "fundamentals_analyst")
    workflow.add_edge("sentiment_corpus_retrieval", "sentiment_analyst")
    workflow.add_edge("risk_corpus_retrieval", "risk_analyst")
    workflow.add_edge("fundamentals_corpus_retrieval", "aggregate_evidence")
    workflow.add_edge("sentiment_corpus_retrieval", "aggregate_evidence")
    workflow.add_edge("risk_corpus_retrieval", "aggregate_evidence")
    workflow.add_edge("fundamentals_analyst", "contradiction_check")
    workflow.add_edge("sentiment_analyst", "contradiction_check")
    workflow.add_edge("risk_analyst", "contradiction_check")
    workflow.add_edge("contradiction_check", "contradiction_review")
    workflow.add_edge("contradiction_review", "thesis_preparation")
    workflow.add_edge("thesis_preparation", "thesis")
    workflow.add_edge("aggregate_evidence", "thesis")
    workflow.add_edge("thesis", "verifier")
    workflow.add_edge("verifier", END)

    return workflow.compile()

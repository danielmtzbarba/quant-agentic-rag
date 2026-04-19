from __future__ import annotations

from stock_agent_rag.evaluation import (
    evaluate_release_gates,
    load_golden_set,
    precision_at_k,
    recall_at_k,
)


def test_default_golden_set_has_expected_coverage() -> None:
    manifest = load_golden_set()
    sectors = {case.sector for case in manifest.cases}
    pairs = {(case.ticker, case.question) for case in manifest.cases}

    assert 20 <= len(manifest.cases) <= 30
    assert len(sectors) >= 3
    assert len(pairs) == len(manifest.cases)


def test_evaluate_release_gates_aggregates_grounding_and_news_quality_metrics() -> None:
    manifest = load_golden_set()
    nvda_case = next(case for case in manifest.cases if case.ticker == "NVDA")
    aapl_case = next(case for case in manifest.cases if case.ticker == "AAPL")

    evaluation = evaluate_release_gates(
        manifest=manifest.model_copy(update={"cases": [nvda_case, aapl_case]}),
        results=[
            {
                "ticker": "NVDA",
                "question": nvda_case.question,
                "verification_status": "pass",
                "repair_attempted": True,
                "verification_metrics": {
                    "malformed_citation_count": 0,
                    "prohibited_placeholder_count": 0,
                    "uncited_numeric_claim_count": 0,
                },
                "retrieval_metrics": {
                    "merged_retrieved_count": 4,
                    "off_ticker_evidence_count": 0,
                },
                "contradictions": [{"topic": "demand"}],
            },
            {
                "ticker": "AAPL",
                "question": aapl_case.question,
                "verification_status": "fail",
                "repair_attempted": False,
                "verification_metrics": {
                    "malformed_citation_count": 1,
                    "prohibited_placeholder_count": 0,
                    "uncited_numeric_claim_count": 2,
                },
                "retrieval_metrics": {
                    "merged_retrieved_count": 5,
                    "off_ticker_evidence_count": 1,
                },
                "contradictions": [],
            },
        ],
    )

    assert evaluation["metrics"] == {
        "evaluated_case_count": 2,
        "golden_set_case_count": 2,
        "golden_set_coverage": 1.0,
        "unmatched_run_count": 0,
        "verification_pass_rate": 0.5,
        "citation_format_compliance": 0.5,
        "unsupported_numeric_claim_count": 2,
        "unsupported_numeric_claim_rate": 1.0,
        "off_ticker_evidence_count": 1,
        "off_ticker_evidence_rate": 0.1111,
        "retrieval_labeled_case_count": 0,
        "retrieval_label_coverage": 0.0,
        "precision@5": 0.0,
        "recall@5": 0.0,
        "expected_contradiction_cases": 1,
        "surfaced_contradiction_cases": 1,
        "contradiction_surfacing_rate": 1.0,
        "repair_attempt_count": 1,
        "pass_rate_after_repair": 1.0,
    }
    assert evaluation["gates"]["citation_format_compliance"] is False
    assert evaluation["gates"]["unsupported_numeric_claim_rate"] is False
    assert evaluation["gates"]["off_ticker_evidence_rate"] is False
    assert evaluation["gates"]["contradiction_surfacing_rate"] is True
    assert evaluation["gates"]["pass_rate_after_repair"] is True
    assert evaluation["status"] == "fail"


def test_precision_and_recall_at_k_use_ranked_retrieval_ids() -> None:
    assert precision_at_k(
        retrieved_source_ids=["a", "b", "c"],
        relevant_source_ids=["b", "d"],
        k=2,
    ) == 0.5
    assert recall_at_k(
        retrieved_source_ids=["a", "b", "c"],
        relevant_source_ids=["b", "d"],
        k=2,
    ) == 0.5


def test_evaluate_release_gates_reports_retrieval_precision_and_recall() -> None:
    manifest = load_golden_set()
    nvda_case = next(case for case in manifest.cases if case.ticker == "NVDA").model_copy(
        update={"relevant_source_ids": ["filing-1", "news-1", "transcript-1"]}
    )
    aapl_case = next(case for case in manifest.cases if case.ticker == "AAPL").model_copy(
        update={"relevant_source_ids": ["filing-2"]}
    )

    evaluation = evaluate_release_gates(
        manifest=manifest.model_copy(update={"cases": [nvda_case, aapl_case]}),
        results=[
            {
                "ticker": "NVDA",
                "question": nvda_case.question,
                "verification_status": "pass",
                "retrieved_sources": ["filing-1", "noise-1", "news-1"],
                "verification_metrics": {},
                "retrieval_metrics": {
                    "merged_retrieved_count": 3,
                    "off_ticker_evidence_count": 0,
                },
                "contradictions": [],
            },
            {
                "ticker": "AAPL",
                "question": aapl_case.question,
                "verification_status": "pass",
                "retrieved_sources": ["noise-2", "filing-2"],
                "verification_metrics": {},
                "retrieval_metrics": {
                    "merged_retrieved_count": 2,
                    "off_ticker_evidence_count": 0,
                },
                "contradictions": [],
            },
        ],
        retrieval_k=2,
    )

    assert evaluation["metrics"]["retrieval_labeled_case_count"] == 2
    assert evaluation["metrics"]["retrieval_label_coverage"] == 1.0
    assert evaluation["metrics"]["precision@2"] == 0.5
    assert evaluation["metrics"]["recall@2"] == 0.6667

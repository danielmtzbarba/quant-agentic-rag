from __future__ import annotations

from types import SimpleNamespace

import stock_agent_rag.workflow as workflow_module
from stock_agent_rag.schemas import (
    AnalystFinding,
    AnalystOutput,
    EvidenceRecord,
    FundamentalsSnapshot,
)
from stock_agent_rag.workflow import build_app


class FixtureRouterModel:
    def __init__(self, *, thesis_report: str, repaired_report: str | None = None) -> None:
        self.model_name = "gpt-test"
        self.temperature = 0.0
        self.thesis_report = thesis_report
        self.repaired_report = repaired_report or thesis_report
        self.thesis_calls = 0
        self.repair_calls = 0
        self.verifier_calls = 0

    def invoke(self, messages):
        system_prompt = messages[0].content
        if "research planner" in system_prompt:
            return self._response("plan")
        if system_prompt.startswith("You are the lead research writer performing"):
            self.repair_calls += 1
            return self._response(self.repaired_report)
        if "lead research writer" in system_prompt:
            self.thesis_calls += 1
            return self._response(self.thesis_report)
        if "verification agent" in system_prompt:
            self.verifier_calls += 1
            return self._response("verification review")
        raise AssertionError(f"Unexpected system prompt: {system_prompt}")

    def _response(self, content: str):
        return SimpleNamespace(
            content=content,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            response_metadata={"model_name": "gpt-test"},
        )


class FixtureStructuredModel:
    def invoke(self, messages):
        system_prompt = messages[0].content
        if "fundamentals analyst" in system_prompt:
            parsed = AnalystOutput(
                summary="fundamentals summary",
                findings=[
                    AnalystFinding(
                        finding="Revenue remains strong",
                        evidence_ids=["filing-1"],
                        confidence=0.9,
                        missing_data=[],
                        finding_type="strength",
                    )
                ],
                overall_confidence=0.9,
            )
        elif "market-sentiment analyst" in system_prompt:
            parsed = AnalystOutput(
                summary="sentiment summary",
                findings=[
                    AnalystFinding(
                        finding="Sentiment remains constructive",
                        evidence_ids=["news-1"],
                        confidence=0.8,
                        missing_data=[],
                        finding_type="positive",
                    )
                ],
                overall_confidence=0.8,
            )
        else:
            parsed = AnalystOutput(summary="risk summary", findings=[], overall_confidence=0.7)
        raw = SimpleNamespace(
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            response_metadata={"model_name": "gpt-test"},
        )
        return {"parsed": parsed, "raw": raw}


def _run_canned_fixture(
    *,
    thesis_report: str,
    repaired_report: str | None = None,
) -> tuple[dict, FixtureRouterModel]:
    router_model = FixtureRouterModel(
        thesis_report=thesis_report,
        repaired_report=repaired_report,
    )
    original_get_model = workflow_module._get_model
    original_get_structured_model = workflow_module._get_structured_model
    original_get_settings = workflow_module.get_settings
    original_fetch_fundamentals_snapshot = workflow_module.fetch_fundamentals_snapshot
    original_local_corpus_search = workflow_module.local_corpus_search

    workflow_module._get_model = lambda: router_model
    workflow_module._get_structured_model = lambda: FixtureStructuredModel()
    workflow_module.get_settings = lambda: SimpleNamespace(
        model_name="gpt-test",
        verifier_max_unsupported_findings=0,
        verifier_max_partially_grounded_findings=0,
    )
    workflow_module.fetch_fundamentals_snapshot = lambda ticker: FundamentalsSnapshot(
        ticker=ticker,
        metrics={"revenue_growth": 0.4},
    )
    workflow_module.local_corpus_search = lambda **kwargs: [
        EvidenceRecord(
            source_id="filing-1",
            ticker="NVDA",
            title="10-Q",
            content="Revenue remains strong.",
            document_type="filing",
        ),
        EvidenceRecord(
            source_id="news-1",
            ticker="NVDA",
            title="Reuters",
            content="Sentiment remains constructive.",
            document_type="news",
            publisher="Reuters",
        ),
    ]

    try:
        result = build_app().invoke(
            {
                "ticker": "NVDA",
                "question": "Generate an evidence-backed investment thesis.",
            }
        )
    finally:
        workflow_module._get_model = original_get_model
        workflow_module._get_structured_model = original_get_structured_model
        workflow_module.get_settings = original_get_settings
        workflow_module.fetch_fundamentals_snapshot = original_fetch_fundamentals_snapshot
        workflow_module.local_corpus_search = original_local_corpus_search

    return result, router_model


def test_canned_e2e_fixture_passes_without_repair() -> None:
    result, router_model = _run_canned_fixture(
        thesis_report=(
            "# Thesis\nRevenue remains strong [source:filing-1]. "
            "Sentiment remains constructive [source:news-1]."
        )
    )

    assert result["verification_status"] == "pass"
    assert bool(result.get("repair_attempted", False)) is False
    assert "[source:filing-1]" in result["report"]
    assert "[source:news-1]" in result["report"]
    assert result["verification_metrics"]["unsupported_findings"] == 0
    assert result["verification_metrics"]["partially_grounded_findings"] == 0
    assert router_model.repair_calls == 0


def test_canned_e2e_fixture_passes_after_single_repair() -> None:
    result, router_model = _run_canned_fixture(
        thesis_report="# Thesis\nRevenue remains strong [source:filing-1].",
        repaired_report=(
            "# Thesis\nRevenue remains strong [source:filing-1]. "
            "Sentiment remains constructive [source:news-1]."
        ),
    )

    assert result["verification_status"] == "pass"
    assert result["repair_attempted"] is True
    assert result["initial_report"] == "# Thesis\nRevenue remains strong [source:filing-1]."
    assert "[source:news-1]" in result["report"]
    assert result["verification_metrics"]["unsupported_findings"] == 0
    assert result["verification_metrics"]["partially_grounded_findings"] == 0
    assert router_model.thesis_calls == 1
    assert router_model.repair_calls == 1
    assert router_model.verifier_calls == 2

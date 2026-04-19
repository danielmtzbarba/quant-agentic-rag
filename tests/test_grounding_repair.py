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


class StubRouterModel:
    def __init__(self) -> None:
        self.model_name = "gpt-test"
        self.temperature = 0.0
        self.thesis_calls = 0
        self.repair_calls = 0
        self.verifier_calls = 0

    def invoke(self, messages):
        system_prompt = messages[0].content
        if "research planner" in system_prompt:
            return self._response("plan")
        if system_prompt.startswith("You are the lead research writer performing"):
            self.repair_calls += 1
            return self._response(
                "# Thesis\nRevenue remains strong [source:filing-1] and sentiment remains "
                "constructive [source:news-1]."
            )
        if "lead research writer" in system_prompt:
            self.thesis_calls += 1
            return self._response("# Thesis\nRevenue remains strong [source:filing-1].")
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


class StubStructuredModel:
    def invoke(self, messages):
        system_prompt = messages[0].content
        if "fundamentals analyst" in system_prompt:
            parsed = AnalystOutput(
                summary="fundamentals summary",
                findings=[
                    AnalystFinding(
                        finding="Revenue remains strong and sentiment is constructive",
                        evidence_ids=["filing-1", "news-1"],
                        confidence=0.9,
                        missing_data=[],
                        finding_type="strength",
                    )
                ],
                overall_confidence=0.9,
            )
        else:
            parsed = AnalystOutput(summary="empty", findings=[], overall_confidence=0.6)
        raw = SimpleNamespace(
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            response_metadata={"model_name": "gpt-test"},
        )
        return {"parsed": parsed, "raw": raw}


def test_workflow_runs_single_repair_pass_and_recovers() -> None:
    router_model = StubRouterModel()
    original_get_model = workflow_module._get_model
    original_get_structured_model = workflow_module._get_structured_model
    original_get_settings = workflow_module.get_settings
    original_fetch_fundamentals_snapshot = workflow_module.fetch_fundamentals_snapshot
    original_local_corpus_search = workflow_module.local_corpus_search

    workflow_module._get_model = lambda: router_model
    workflow_module._get_structured_model = lambda: StubStructuredModel()
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

    assert result["verification_status"] == "pass"
    assert result["repair_attempted"] is True
    assert result["initial_report"] == "# Thesis\nRevenue remains strong [source:filing-1]."
    assert "[source:news-1]" in result["report"]
    assert router_model.thesis_calls == 1
    assert router_model.repair_calls == 1
    assert router_model.verifier_calls == 2

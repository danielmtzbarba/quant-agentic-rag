from __future__ import annotations

import stock_agent_rag.service as service_module
from stock_agent_rag.schemas import ResearchRequest
from stock_agent_rag.service import ResearchService


class StubGraph:
    def invoke(self, payload: dict) -> dict:
        return {
            "plan": f"plan for {payload['ticker']}",
            "report": "report body",
            "verification_summary": "verified",
            "node_metrics": {
                "planner": {
                    "model_name": "gpt-test",
                    "provider": "openai",
                    "temperature": 0.0,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "retry_count": 1,
                    "timeout_count": 0,
                    "estimated_cost_usd": 0.000123,
                    "latency_ms": 12.0,
                }
            },
            "fundamentals_evidence": [],
            "sentiment_evidence": [],
            "risk_evidence": [],
            "retrieved_evidence": [],
        }


def test_research_service_maps_graph_output() -> None:
    original = service_module.get_settings
    service_module.get_settings = lambda: type("StubSettings", (), {"db_enabled": False})()

    try:
        service = ResearchService(app=StubGraph())
        result = service.run(ResearchRequest(ticker="msft", question="What is the thesis?"))
    finally:
        service_module.get_settings = original

    assert result.ticker == "MSFT"
    assert result.plan == "plan for MSFT"
    assert result.report == "report body"
    assert result.verification_status == "unknown"
    assert result.verification_summary == "verified"
    assert result.token_usage == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }
    assert result.model_metadata == {
        "models": ["gpt-test"],
        "providers": ["openai"],
        "temperatures": [0.0],
    }
    assert result.runtime_metrics == {"retry_count": 1, "timeout_count": 0}
    assert result.retrieval_metrics["merged_retrieved_count"] == 0
    assert result.estimated_cost_usd == 0.000123

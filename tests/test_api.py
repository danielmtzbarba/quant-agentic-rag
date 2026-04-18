from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import stock_agent_rag.api as api_module
from stock_agent_rag.api import create_app
from stock_agent_rag.schemas import ResearchResponse


class StubResearchService:
    def run(self, request):
        return ResearchResponse(
            ticker=request.ticker.upper(),
            question=request.question,
            plan="test plan",
            report="test report",
            verification_summary="pass",
            retrieved_sources=["source-1"],
            latency_ms=12.5,
        )


@pytest.mark.asyncio
async def test_healthz_returns_ok() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=create_app()),
        base_url="http://test",
    ) as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_research_endpoint_returns_workflow_response() -> None:
    app = create_app()
    original = api_module.get_research_service
    api_module.get_research_service = lambda: StubResearchService()

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/research",
                json={
                    "ticker": "nvda",
                    "question": "Generate an evidence-backed investment thesis.",
                },
            )
    finally:
        api_module.get_research_service = original

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "NVDA"
    assert payload["retrieved_sources"] == ["source-1"]
    assert response.headers["x-request-id"]

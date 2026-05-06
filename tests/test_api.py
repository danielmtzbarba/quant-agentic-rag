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


def _stub_settings(**overrides):
    values = {
        "app_env": "local",
        "app_name": "stock-agent-rag",
        "log_level": "INFO",
        "resolved_log_format": "json",
        "cors_origins": [],
        "openai_api_key": "sk-test-key",
        "db_enabled": False,
        "research_service_auth_token": None,
    }
    values.update(overrides)
    return type("StubSettings", (), values)()


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
async def test_readyz_returns_ok_when_llm_is_configured_and_db_is_disabled() -> None:
    original_get_settings = api_module.get_settings
    original_check_database_connection = api_module.check_database_connection

    api_module.get_settings = lambda: _stub_settings()
    api_module.check_database_connection = lambda: True

    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()),
            base_url="http://test",
        ) as client:
            response = await client.get("/readyz")
    finally:
        api_module.get_settings = original_get_settings
        api_module.check_database_connection = original_check_database_connection

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "local",
        "llm": "configured",
        "database": "disabled",
    }


@pytest.mark.asyncio
async def test_readyz_returns_503_when_llm_api_key_is_missing() -> None:
    original_get_settings = api_module.get_settings
    original_check_database_connection = api_module.check_database_connection

    api_module.get_settings = lambda: _stub_settings(
        app_env="test",
        openai_api_key=None,
    )
    api_module.check_database_connection = lambda: True

    try:
        async with AsyncClient(
            transport=ASGITransport(app=create_app()),
            base_url="http://test",
        ) as client:
            response = await client.get("/readyz")
    finally:
        api_module.get_settings = original_get_settings
        api_module.check_database_connection = original_check_database_connection

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "environment": "test",
        "llm": "missing_api_key",
        "database": "disabled",
    }


@pytest.mark.asyncio
async def test_research_endpoint_requires_bearer_token_when_configured() -> None:
    original_get_settings = api_module.get_settings
    original_get_research_service = api_module.get_research_service
    api_module.get_settings = lambda: _stub_settings(research_service_auth_token="secret-token")
    api_module.get_research_service = lambda: StubResearchService()

    try:
        app = create_app()
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
        api_module.get_settings = original_get_settings
        api_module.get_research_service = original_get_research_service

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing bearer token."


@pytest.mark.asyncio
async def test_research_endpoint_accepts_bearer_token_when_configured() -> None:
    original_get_settings = api_module.get_settings
    original_get_research_service = api_module.get_research_service
    api_module.get_settings = lambda: _stub_settings(research_service_auth_token="secret-token")
    api_module.get_research_service = lambda: StubResearchService()

    try:
        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/research",
                headers={"Authorization": "Bearer secret-token"},
                json={
                    "ticker": "nvda",
                    "question": "Generate an evidence-backed investment thesis.",
                },
            )
    finally:
        api_module.get_settings = original_get_settings
        api_module.get_research_service = original_get_research_service

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_research_endpoint_returns_workflow_response() -> None:
    original_get_settings = api_module.get_settings
    original_get_research_service = api_module.get_research_service
    api_module.get_settings = lambda: _stub_settings(research_service_auth_token=None)
    api_module.get_research_service = lambda: StubResearchService()

    try:
        app = create_app()
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
        api_module.get_settings = original_get_settings
        api_module.get_research_service = original_get_research_service

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "NVDA"
    assert payload["retrieved_sources"] == ["source-1"]
    assert response.headers["x-request-id"]

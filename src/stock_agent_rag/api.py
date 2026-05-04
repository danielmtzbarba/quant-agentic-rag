from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .db import check_database_connection
from .logging import get_logger, setup_logging
from .middleware import register_cors, register_middleware
from .schemas import HealthResponse, ReadinessResponse, ResearchRequest, ResearchResponse
from .service import ResearchService, get_research_service

logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level, settings.resolved_log_format)
    app = FastAPI(title="Stock Agent RAG", version="0.1.0")
    register_cors(app, origins=settings.cors_origins)
    register_middleware(app)
    logger.info(
        "application configured",
        extra={
            "app_env": settings.app_env,
            "app_name": settings.app_name,
            "log_format": settings.resolved_log_format,
        },
    )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok", environment=settings.app_env)

    @app.get("/readyz", response_model=ReadinessResponse)
    async def readyz() -> JSONResponse:
        llm_status: str = "configured" if settings.openai_api_key else "missing_api_key"
        database_status: str = "disabled"

        if settings.db_enabled:
            try:
                await run_in_threadpool(check_database_connection)
                database_status = "ok"
            except Exception:
                logger.warning("database readiness check failed", exc_info=True)
                database_status = "unavailable"

        ready = llm_status == "configured" and database_status in {"disabled", "ok"}
        payload = ReadinessResponse(
            status="ok" if ready else "degraded",
            environment=settings.app_env,
            llm=llm_status,
            database=database_status,
        )
        return JSONResponse(
            status_code=200 if ready else 503,
            content=payload.model_dump(mode="json"),
        )

    @app.post("/v1/research", response_model=ResearchResponse)
    async def run_research(request: ResearchRequest) -> ResearchResponse:
        service: ResearchService = get_research_service()
        logger.info(
            "research request received",
            extra={"ticker": request.ticker.upper(), "question": request.question},
        )
        return await run_in_threadpool(service.run, request)

    return app

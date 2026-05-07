from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .db import check_database_connection
from .logging import get_logger, set_service_name, setup_logging
from .metrics import (
    RESEARCH_REQUEST_LATENCY,
    RESEARCH_REQUESTS_TOTAL,
    mark_successful_research_request,
    render_metrics_response,
)
from .middleware import register_cors, register_middleware
from .schemas import HealthResponse, ReadinessResponse, ResearchRequest, ResearchResponse
from .service import ResearchService, get_research_service

logger = get_logger(__name__)


def _verify_internal_bearer(authorization: str | None, *, expected_token: str | None) -> None:
    token = (expected_token or "").strip()
    if not token:
        return
    header_value = (authorization or "").strip()
    if not header_value.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    provided_token = header_value.removeprefix("Bearer ").strip()
    if provided_token != token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        )


def create_app() -> FastAPI:
    settings = get_settings()
    set_service_name(settings.app_name)
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

    @app.get("/metrics")
    async def metrics():
        return render_metrics_response()

    @app.post("/v1/research", response_model=ResearchResponse)
    async def run_research(
        request: ResearchRequest,
        authorization: str | None = Header(default=None),
    ) -> ResearchResponse:
        _verify_internal_bearer(
            authorization,
            expected_token=settings.research_service_auth_token,
        )
        service: ResearchService = get_research_service()
        logger.info(
            "research request received",
            extra={"ticker": request.ticker.upper(), "question": request.question},
        )
        status_label = "error"
        start = perf_counter()
        try:
            response = await run_in_threadpool(service.run, request)
            status_label = "success"
            mark_successful_research_request()
            elapsed_seconds = perf_counter() - start
            logger.info(
                "research_request_completed",
                extra={
                    "ticker": response.ticker,
                    "question": request.question,
                    "verification_status": response.verification_status,
                    "retrieved_source_count": len(response.retrieved_sources),
                    "latency_ms": round(elapsed_seconds * 1000, 2),
                    "estimated_cost_usd": response.estimated_cost_usd,
                    "total_tokens": (response.token_usage or {}).get("total_tokens"),
                },
            )
            return response
        except Exception:
            logger.exception(
                "research_request_failed",
                extra={
                    "ticker": request.ticker.upper(),
                    "question": request.question,
                },
            )
            raise
        finally:
            RESEARCH_REQUESTS_TOTAL.labels(status=status_label).inc()
            RESEARCH_REQUEST_LATENCY.observe(perf_counter() - start)

    return app

from __future__ import annotations

from fastapi import FastAPI

from .config import get_settings
from .logging import get_logger, setup_logging
from .middleware import register_middleware
from .schemas import HealthResponse, ResearchRequest, ResearchResponse
from .service import ResearchService, get_research_service

logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level, settings.resolved_log_format)
    app = FastAPI(title="Stock Agent RAG", version="0.1.0")
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

    @app.post("/v1/research", response_model=ResearchResponse)
    async def run_research(request: ResearchRequest) -> ResearchResponse:
        service: ResearchService = get_research_service()
        logger.info(
            "research request received",
            extra={"ticker": request.ticker.upper(), "question": request.question},
        )
        return service.run(request)

    return app

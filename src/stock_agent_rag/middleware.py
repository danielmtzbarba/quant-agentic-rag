from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .logging import get_logger, reset_request_id, set_request_id

logger = get_logger(__name__)


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid4()))
        token = set_request_id(request_id)
        start = perf_counter()
        response = await call_next(request)
        latency_ms = round((perf_counter() - start) * 1000, 2)
        logger.info(
            "http request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "client": request.client.host if request.client else None,
            },
        )
        response.headers["x-request-id"] = request_id
        reset_request_id(token)
        return response


def register_cors(app: FastAPI, *, origins: list[str]) -> None:
    if not origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

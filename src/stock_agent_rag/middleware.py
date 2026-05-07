from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .logging import get_logger, reset_request_id, set_request_id
from .metrics import (
    HTTP_4XX_ERRORS,
    HTTP_5XX_ERRORS,
    HTTP_REQUEST_LATENCY,
    HTTP_REQUESTS_TOTAL,
)

logger = get_logger(__name__)


DEBUG_PATHS = {
    "/healthz",
    "/readyz",
    "/metrics",
}
DEBUG_PREFIXES = ()
QUIET_PATHS = set()
SLOW_REQUEST_SECONDS = 1.0


def _request_path_label(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return "__unmatched__"


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid4()))
        token = set_request_id(request_id)
        start = perf_counter()
        path = _request_path_label(request)
        is_noisy = path in DEBUG_PATHS or any(path.startswith(prefix) for prefix in DEBUG_PREFIXES)
        is_quiet = path in QUIET_PATHS
        logger.debug(
            "request_started",
            extra={
                "method": request.method,
                "path": path,
                "client": request.client.host if request.client else None,
            },
        )
        response = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            logger.exception(
                "request_unhandled_exception",
                extra={
                    "method": request.method,
                    "path": path,
                    "client": request.client.host if request.client else None,
                },
            )
            raise
        finally:
            latency_seconds = perf_counter() - start
            latency_ms = round(latency_seconds * 1000, 2)
            path = _request_path_label(request)
            status_class = f"{status_code // 100}xx"
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                path=path,
                status_class=status_class,
            ).inc()
            HTTP_REQUEST_LATENCY.labels(
                method=request.method,
                path=path,
                status_class=status_class,
            ).observe(latency_seconds)
            if 400 <= status_code < 500:
                HTTP_4XX_ERRORS.labels(method=request.method, path=path).inc()
            elif status_code >= 500:
                HTTP_5XX_ERRORS.labels(method=request.method, path=path).inc()
            if status_code >= 500:
                log_method = logger.error
            elif status_code >= 400:
                log_method = logger.warning
            elif latency_seconds >= SLOW_REQUEST_SECONDS:
                log_method = logger.info
            elif is_noisy or is_quiet:
                log_method = logger.debug
            else:
                log_method = logger.info
            log_method(
                "request_finished",
                extra={
                    "method": request.method,
                    "path": path,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "client": request.client.host if request.client else None,
                },
            )
            if response is not None:
                response.headers["x-request-id"] = request_id
            reset_request_id(token)


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

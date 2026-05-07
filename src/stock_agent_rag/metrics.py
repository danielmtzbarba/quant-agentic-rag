from __future__ import annotations

from time import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

HTTP_REQUESTS_TOTAL = Counter(
    "research_http_requests_total",
    "Total number of HTTP requests handled by the research service",
    ["method", "path", "status_class"],
)

HTTP_4XX_ERRORS = Counter(
    "research_http_4xx_errors_total",
    "Total number of 4xx responses returned by the research service",
    ["method", "path"],
)

HTTP_5XX_ERRORS = Counter(
    "research_http_5xx_errors_total",
    "Total number of 5xx responses returned by the research service",
    ["method", "path"],
)

HTTP_REQUEST_LATENCY = Histogram(
    "research_http_request_latency_seconds",
    "Latency of HTTP requests handled by the research service",
    ["method", "path", "status_class"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

RESEARCH_REQUESTS_TOTAL = Counter(
    "research_requests_total",
    "Total number of research workflow executions",
    ["status"],
)

RESEARCH_REQUEST_LATENCY = Histogram(
    "research_request_latency_seconds",
    "Latency of completed research workflow executions",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

LAST_SUCCESSFUL_RESEARCH_REQUEST_TIMESTAMP = Gauge(
    "research_last_successful_request_timestamp_seconds",
    "Unix timestamp of the last successful research workflow execution",
)


def mark_successful_research_request() -> None:
    LAST_SUCCESSFUL_RESEARCH_REQUEST_TIMESTAMP.set(time())


def render_metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

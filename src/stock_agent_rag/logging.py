from __future__ import annotations

import contextvars
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id",
    default=None,
)
_RICH_CONSOLE = Console(stderr=False, soft_wrap=True)
_RESERVED_ATTRS = set(logging.makeLogRecord({}).__dict__.keys())
_IGNORED_EXTRA_FIELDS = {"color_message"}


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _serialize(value: object) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(char.isspace() for char in text) or any(char in text for char in '"='):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    extras: dict[str, Any] = {}

    request_id = request_id_var.get()
    if request_id:
        extras["request_id"] = request_id

    for key, value in record.__dict__.items():
        if (
            key.startswith("_")
            or key in _RESERVED_ATTRS
            or key in _IGNORED_EXTRA_FIELDS
            or value is None
        ):
            continue
        extras[key] = value

    return extras


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        parts = [
            f"ts={_utc_timestamp()}",
            f"level={record.levelname.lower()}",
            f"logger={record.name}",
            f"event={_serialize(record.getMessage())}",
        ]

        for key, value in sorted(_extra_fields(record).items()):
            parts.append(f"{key}={_serialize(value)}")

        if record.exc_info:
            parts.append(f"exc_info={_serialize(self.formatException(record.exc_info))}")

        return " ".join(parts)


class RichConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = record.getMessage()
        extras = _extra_fields(record)
        if not extras:
            return base

        suffix = " ".join(f"{key}={_serialize(value)}" for key, value in sorted(extras.items()))
        return f"{base} [{suffix}]"


def _build_handler(log_format: str) -> logging.Handler:
    if log_format == "rich":
        handler = RichHandler(
            console=_RICH_CONSOLE,
            rich_tracebacks=True,
            show_path=False,
            markup=False,
            show_time=True,
            omit_repeated_times=False,
            log_time_format="%H:%M:%S",
        )
        handler.setFormatter(RichConsoleFormatter())
        return handler

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    return handler


def _configure_named_logger(
    name: str,
    level: str,
    log_format: str,
    *,
    propagate: bool,
) -> None:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = propagate
    logger.setLevel(level.upper())

    if not propagate:
        logger.addHandler(_build_handler(log_format))


def setup_logging(level: str = "INFO", log_format: str = "logfmt") -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.propagate = False

    effective_root_format = "rich" if log_format == "hybrid" else log_format
    handler = _build_handler(effective_root_format)
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())

    if log_format == "hybrid":
        _configure_named_logger("uvicorn.access", level, "logfmt", propagate=False)
        _configure_named_logger("httpx", level, "logfmt", propagate=False)
    else:
        _configure_named_logger("uvicorn.access", level, effective_root_format, propagate=True)
        _configure_named_logger("httpx", level, effective_root_format, propagate=True)

    _configure_named_logger("uvicorn.error", level, effective_root_format, propagate=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_request_id(request_id: str | None) -> contextvars.Token[str | None]:
    return request_id_var.set(request_id)


def reset_request_id(token: contextvars.Token[str | None]) -> None:
    request_id_var.reset(token)

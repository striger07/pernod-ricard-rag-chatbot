"""Structured logging without secrets."""

import logging
import sys
from typing import Any, MutableMapping, Optional

import structlog

from config.settings import Settings, get_settings

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
        "grok_api_key",
        "openai_api_key",
        "qdrant_api_key",
        "admin_api_token",
    }
)


def _drop_sensitive(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    redacted: MutableMapping[str, Any] = {}
    for key, value in event_dict.items():
        if key.lower() in _SENSITIVE_KEYS or any(
            part in key.lower() for part in ("api_key", "token", "password", "secret")
        ):
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = value
    return redacted


def configure_logging(settings: Optional[Settings] = None) -> None:
    """Configure stdlib + structlog once per process."""
    active = settings or get_settings()
    level = getattr(logging, active.log_level, logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _drop_sensitive,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

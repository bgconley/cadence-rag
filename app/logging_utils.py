from __future__ import annotations

import contextvars
import logging
from typing import Optional

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)
_configured = False


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _request_id_var.get("-")
        return True


def configure_logging(level: str = "INFO") -> None:
    global _configured
    root = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if not _configured:
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s [%(name)s] [req=%(request_id)s] %(message)s"
                )
            )
            root.addHandler(handler)
        for handler in root.handlers:
            handler.addFilter(RequestIdFilter())
        _configured = True

    root.setLevel(numeric_level)


def set_request_id(request_id: str) -> contextvars.Token[str]:
    return _request_id_var.set(request_id)


def reset_request_id(token: contextvars.Token[str]) -> None:
    _request_id_var.reset(token)


def get_request_id() -> str:
    return _request_id_var.get("-")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)

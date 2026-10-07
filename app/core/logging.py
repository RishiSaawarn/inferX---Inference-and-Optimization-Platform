import json
import logging
from typing import Any

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Add any extra fields passed in the log record's __dict__
        # Specifically handling the 'extra' dict passed to logger methods
        # However, standard logging puts extra keys directly into the record dict.
        # We will extract keys that are not standard LogRecord keys.

        standard_keys = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
        }

        for key, value in record.__dict__.items():
            if key not in standard_keys:
                log_record[key] = value

        return json.dumps(log_record)


def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(settings.log_level.upper())

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    formatter = JSONFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Configure uvicorn loggers to use our formatter
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
        l = logging.getLogger(logger_name)
        l.handlers = [handler]
        l.propagate = False

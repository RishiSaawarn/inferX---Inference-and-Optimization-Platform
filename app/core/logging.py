import logging
import json
import traceback
import contextvars
from datetime import datetime, timezone
import uuid
from typing import Any

request_id_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx_var.get()
        }
        
        # P3-2: Handle extra args safely
        if hasattr(record, "status"):
            log_record["status"] = record.status
        if hasattr(record, "env"):
            log_record["env"] = record.env
            
        # P3-2: Fix traceback logging in JSON
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        elif record.exc_text:
            log_record["exc_info"] = record.exc_text

        # P3-2: stringify defaults to avoid unserializable objects crashing the logger
        return json.dumps(log_record, default=str)

def setup_logging(level: int = logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    
    # Silence third party noisy logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)

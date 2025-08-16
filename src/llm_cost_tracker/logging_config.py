"""Structured logging configuration for LLM Cost Tracker."""

import json
import logging
import logging.config
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict

# Context variables for request tracing
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request context if available
        if request_id := request_id_var.get():
            log_data["request_id"] = request_id

        if user_id := user_id_var.get():
            log_data["user_id"] = user_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs."""

    SENSITIVE_PATTERNS = [
        "password",
        "secret",
        "token",
        "key",
        "api_key",
        "authorization",
        "credential",
        "x-api-key",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out sensitive information from log records."""
        # Sanitize message
        message = record.getMessage().lower()
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message:
                record.msg = "[REDACTED - Contains sensitive information]"
                record.args = ()
                break

        return True


def configure_logging(log_level: str = "INFO", structured: bool = True) -> None:
    """Configure application logging."""

    # Base logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": StructuredFormatter,
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "filters": {
            "security": {
                "()": SecurityFilter,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "structured" if structured else "simple",
                "filters": ["security"],
                "level": log_level,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/llm-cost-tracker.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "formatter": "structured" if structured else "simple",
                "filters": ["security"],
                "level": log_level,
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/llm-cost-tracker-error.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "formatter": "structured" if structured else "simple",
                "filters": ["security"],
                "level": "ERROR",
            },
        },
        "loggers": {
            "llm_cost_tracker": {
                "handlers": ["console", "file", "error_file"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "fastapi": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": log_level,
        },
    }

    # Ensure log directory exists
    import os

    os.makedirs("logs", exist_ok=True)

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(f"llm_cost_tracker.{name}")


def set_request_context(request_id: str = None, user_id: str = None) -> None:
    """Set request context for logging."""
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


class LogMetrics:
    """Collect and report logging metrics."""

    def __init__(self):
        self.error_count = 0
        self.warning_count = 0
        self.info_count = 0
        self.debug_count = 0

    def increment_error(self):
        """Increment error count."""
        self.error_count += 1

    def increment_warning(self):
        """Increment warning count."""
        self.warning_count += 1

    def increment_info(self):
        """Increment info count."""
        self.info_count += 1

    def increment_debug(self):
        """Increment debug count."""
        self.debug_count += 1

    def get_metrics(self) -> Dict[str, int]:
        """Get current logging metrics."""
        return {
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "debug_count": self.debug_count,
            "total_logs": (
                self.error_count
                + self.warning_count
                + self.info_count
                + self.debug_count
            ),
        }


# Global metrics instance
log_metrics = LogMetrics()


class MetricsHandler(logging.Handler):
    """Custom handler to collect logging metrics."""

    def emit(self, record: logging.LogRecord) -> None:
        """Count log records by level."""
        if record.levelno >= logging.ERROR:
            log_metrics.increment_error()
        elif record.levelno >= logging.WARNING:
            log_metrics.increment_warning()
        elif record.levelno >= logging.INFO:
            log_metrics.increment_info()
        else:
            log_metrics.increment_debug()

"""
Logging configuration for Ijon PDF RAG System.

This module sets up structured logging with both console and file outputs,
including context tracking and performance metrics.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler

from src.config import get_settings


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """Add context information to log records."""

    def __init__(self) -> None:
        """Initialize the context filter."""
        super().__init__()
        self.context: dict[str, Any] = {}

    def set_context(self, **kwargs: Any) -> None:
        """Set context values."""
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context values."""
        self.context.clear()

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to the log record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


# Global context filter instance
context_filter = ContextFilter()


def setup_logging(
    log_level: Optional[str] = None,
    log_file_path: Optional[Path] = None,
    use_structured_logging: bool = True,
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (defaults to settings)
        log_file_path: Path to log file (defaults to settings)
        use_structured_logging: Use JSON structured logging for files
    """
    settings = get_settings()
    
    # Use provided values or fall back to settings
    log_level = log_level or settings.log_level
    log_file_path = log_file_path or settings.get_log_file_path()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with Rich formatting
    console_handler = RichHandler(
        console=Console(stderr=True),
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=settings.dev_mode,
    )
    console_handler.setLevel(log_level)
    console_handler.addFilter(context_filter)
    
    # Simple format for console
    console_formatter = logging.Formatter(
        "%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if log file path is provided
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.addFilter(context_filter)
        
        # Use structured or simple formatter based on preference
        if use_structured_logging:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
        
        root_logger.addHandler(file_handler)

    # Configure third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "log_level": log_level,
            "log_file": str(log_file_path) if log_file_path else None,
            "structured_logging": use_structured_logging,
        },
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary logging context."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with context values."""
        self.context = kwargs
        self.old_context: dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        """Enter the context and set values."""
        # Save old values
        for key in self.context:
            if hasattr(context_filter.context, key):
                self.old_context[key] = context_filter.context[key]
        
        # Set new values
        context_filter.set_context(**self.context)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and restore old values."""
        # Remove added context
        for key in self.context:
            if key in context_filter.context:
                del context_filter.context[key]
        
        # Restore old values
        if self.old_context:
            context_filter.set_context(**self.old_context)


def log_performance(func):
    """
    Decorator to log function performance.
    
    Usage:
        @log_performance
        def process_pdf(file_path: str) -> None:
            ...
    """
    import functools
    import time
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        """Async wrapper for performance logging."""
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            with LogContext(function=func.__name__):
                logger.debug(f"Starting {func.__name__}")
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"Completed {func.__name__}",
                    extra={"duration_seconds": duration},
                )
                return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Failed {func.__name__}",
                extra={"duration_seconds": duration, "error": str(e)},
            )
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        """Sync wrapper for performance logging."""
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            with LogContext(function=func.__name__):
                logger.debug(f"Starting {func.__name__}")
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"Completed {func.__name__}",
                    extra={"duration_seconds": duration},
                )
                return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Failed {func.__name__}",
                extra={"duration_seconds": duration, "error": str(e)},
            )
            raise
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# Initialize logging on import if not already done
if not logging.getLogger().handlers:
    setup_logging()
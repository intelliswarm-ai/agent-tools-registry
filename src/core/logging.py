import logging
import sys
from typing import Any, Dict

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentation.
    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def format_record(record: Dict[str, Any]) -> str:
    """
    Custom format for loguru loggers.
    Uses a custom format string that includes:
    - timestamp
    - level
    - module and function
    - message
    - any extra fields as JSON
    """
    format_string = "[{time:YYYY-MM-DD HH:mm:ss}] [{level}] [{module}:{function}] {message}"

    if record["extra"]:
        format_string += "\nExtra: {extra}"

    return format_string


def setup_logging() -> None:
    """
    Configures logging using loguru with a custom format and handlers.
    - Removes default loguru handler
    - Adds custom formatted handler to stdout
    - Adds custom formatted handler to file
    - Intercepts standard logging
    """
    # Remove default handler
    logger.remove()

    # Add stdout handler
    logger.add(
        sys.stdout,
        enqueue=True,
        backtrace=True,
        format=format_record,
        level="INFO",
    )

    # Add file handler
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        enqueue=True,
        backtrace=True,
        format=format_record,
        level="INFO",
    )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Disable uvicorn access logging
    logging.getLogger("uvicorn.access").disabled = True

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)

def log_request_details(request_data: Dict[str, Any], logger: logging.Logger) -> None:
    """Log details about an incoming request."""
    logger.info(f"Request data: {request_data}")

def log_response_details(response_data: Dict[str, Any], logger: logging.Logger) -> None:
    """Log details about an outgoing response."""
    logger.info(f"Response data: {response_data}")

def log_error(error: Exception, logger: logging.Logger) -> None:
    """Log error details."""
    logger.error(f"Error occurred: {str(error)}", exc_info=True) 
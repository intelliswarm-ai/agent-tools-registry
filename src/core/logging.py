import logging
import sys
from typing import Any, Dict, Optional
from pathlib import Path

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
    Configures logging with consistent formatting across all loggers.
    - Sets up root logger
    - Configures console and file handlers
    - Ensures log directory exists
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # Disable uvicorn access logging
    logging.getLogger("uvicorn.access").disabled = True

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with proper configuration.
    Logger will inherit configuration from root logger.
    
    Args:
        name: Optional name for the logger
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name or __name__)

def log_request_details(request_data: Dict[str, Any], logger: logging.Logger) -> None:
    """Log details about an incoming request."""
    logger.info(f"Request data: {request_data}")

def log_response_details(response_data: Dict[str, Any], logger: logging.Logger) -> None:
    """Log details about an outgoing response."""
    logger.info(f"Response data: {response_data}")

def log_error(error: Exception, logger: logging.Logger) -> None:
    """Log error details with full traceback."""
    logger.error(f"Error occurred: {str(error)}", exc_info=True) 
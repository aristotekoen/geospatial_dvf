"""
Logging configuration for the DVF pipeline.

Usage:
    from utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.warning("Missing data")
    logger.error("Failed to download")
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Optional


def setup_logger(
    name: str = "dvf_pipeline",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format: simple for console
    console_format = logging.Formatter(
        fmt="%(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "dvf_pipeline") -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If no handlers, set up with defaults
    if not logger.handlers and not logger.parent.handlers:
        setup_logger(name)
    
    return logger


# Convenience functions for step headers
def log_step_header(logger: logging.Logger, step_num: int, title: str) -> None:
    """Log a formatted step header."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP {step_num}: {title}")
    logger.info("=" * 60)


def log_step_complete(logger: logging.Logger, message: str) -> None:
    """Log step completion with checkmark."""
    logger.info(f"\n✅ {message}")


def log_section(logger: logging.Logger, title: str) -> None:
    """Log a section header."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


@contextmanager
def log_timed(logger: logging.Logger, task_name: str):
    """Context manager to log the duration of a task.
    
    Usage:
        with log_timed(logger, "Processing data"):
            process_data()
        # Logs: "Processing data completed in 5.2s"
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"  ⏱️  {task_name} completed in {format_duration(elapsed)}")

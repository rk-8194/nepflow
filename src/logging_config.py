"""Logging configuration for NEPFlow."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    project_name: str,
    log_dir: Optional[Path] = None,
    debug: bool = False,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging for a project.
    
    Creates a logger with both console and file handlers.
    All nepflow.* loggers will inherit these handlers through propagation.
    
    Args:
        project_name: Name of the project for log file naming
        log_dir: Directory for log files (default: ./logs)
        debug: Enable debug logging (sets level to DEBUG)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    if debug:
        level = logging.DEBUG
    
    # Create root logger for all nepflow modules
    root_logger = logging.getLogger("nepflow")
    root_logger.setLevel(level)
    
    # Default log directory
    if log_dir is None:
        log_dir = Path(".") / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # File handler
    log_file = log_dir / f"{project_name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Ensure propagation is enabled for child loggers
    root_logger.propagate = True
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)

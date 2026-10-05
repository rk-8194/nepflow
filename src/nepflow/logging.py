"""Central logging configuration for NEPFlow."""

import logging
import logging.handlers
from pathlib import Path


def configure_logging(
    project_name: str,
    log_dir: Path | None = None,
    debug: bool = False,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure the ``nepflow`` logger with console and project-file handlers.

    Repeated configuration replaces and closes the package handlers so a
    process cannot accumulate duplicate output handlers.
    """
    if debug:
        level = logging.DEBUG

    project_logger = logging.getLogger("nepflow")
    project_logger.setLevel(level)

    resolved_log_dir = Path("logs") if log_dir is None else Path(log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        resolved_log_dir / f"{project_name}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    old_handlers = project_logger.handlers[:]
    project_logger.handlers.clear()
    for handler in old_handlers:
        handler.close()

    project_logger.addHandler(console_handler)
    project_logger.addHandler(file_handler)
    project_logger.propagate = True

    # NepTrainKit emits through loguru; keep its internal diagnostics out of
    # the project log when the optional dependency is installed.
    try:
        from loguru import logger as loguru_logger
    except ImportError:
        # loguru integration is optional; stdlib logging remains authoritative.
        project_logger.debug("Optional loguru integration is unavailable")
    else:
        loguru_logger.disable("NepTrainKit")

    return project_logger

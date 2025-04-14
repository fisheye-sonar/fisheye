import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def log_progress(logger, current: int, total: int, prefix: str = "", every: int = 10):
    """
    Logs percentage progress using a given logger.

    Args:
        logger: The logger instance.
        current: Current index in iteration (0-based).
        total: Total number of items.
        prefix: Optional prefix message.
        every: Log only every N percent to reduce noise (default is 5%).
    """
    if total == 0:
        return  # Avoid division by zero

    percent_complete = ((current + 1) / total) * 100
    # Only log at specified interval or at 100%
    if percent_complete % every < 100 / total or percent_complete == 100:
        logger.debug(
            f"{prefix}Progress: {percent_complete:.1f}% ({current + 1}/{total})"
        )


def setup_logging(
    base_log_dir: str = "logs",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    modules: list[str] = None,
    file_logging: bool = False,
) -> None:
    """
    Configure logging across modules.

    Args:
        base_log_dir: Directory to store log files.
        level: Logging level (e.g., logging.INFO, logging.ERROR).
        max_bytes: Max size of each log file before rotation.
        backup_count: Number of rotated log files to keep.
        modules: Optional list of module names to configure separately.
        file_logging: Enable or disable logging to local file(s).
    """

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    if file_logging and modules:
        Path(base_log_dir).mkdir(parents=True, exist_ok=True)
        for module in modules:
            logger = logging.getLogger(f"fisheye.{module}")
            log_path = os.path.join(base_log_dir, f"{module}.log")
            file_handler = RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.setLevel(level)
            logger.addHandler(file_handler)
            logger.propagate = True

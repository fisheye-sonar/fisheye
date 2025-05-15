import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog


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
    file_logging: bool = False,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    job_id: str = None,
):
    shared_processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.FILENAME]
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Configure structlog but no renderer here
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(),
            foreign_pre_chain=shared_processors,
        )
    )
    root_logger.addHandler(console_handler)

    if file_logging:
        Path(base_log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}-{job_id}.log" if job_id else f"fisheye-{timestamp}.log"
        log_path = os.path.join(base_log_dir, filename)

        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=shared_processors,
            )
        )

        root_logger.addHandler(file_handler)

import logging
import os
import queue
import time
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import structlog

progress_queue: ContextVar[Optional[queue.Queue]] = ContextVar(
    "progress_queue", default=None
)


def _route_to_progress_queue(logger, method, event_dict):
    """Fires on every structlog call, routes to progress queue if available."""
    q = progress_queue.get()
    if q is not None:
        try:
            q.put_nowait(event_dict.copy())
        except queue.Full:
            pass
    return event_dict


class ProgressTracker:
    """Logs progress at most once per min_interval seconds, always at 100%.

    Instantiate once per pipeline stage; each instance holds its own timer so
    concurrent jobs never suppress each other's updates.
    """

    def __init__(self, min_interval: float = 5.0):
        self._min_interval = min_interval
        self._last_log: float = 0.0

    def log(self, logger, current: int, total: int, prefix: str = "") -> None:
        if total == 0:
            return
        now = time.monotonic()
        is_done = current + 1 >= total
        if (now - self._last_log) >= self._min_interval or is_done:
            self._last_log = now
            pct = (current + 1) / total * 100
            logger.debug(f"{prefix}Progress: {pct:.1f}% ({current + 1}/{total})")


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
        _route_to_progress_queue,
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

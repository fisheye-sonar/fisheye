"""Logging setup for the FishEye desktop app."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from queue import Queue

import structlog

from fisheye_app.paths import log_dir


class TextQueueHandler(logging.Handler):
    def __init__(self, queue: Queue) -> None:
        super().__init__(level=logging.INFO)
        self._queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put_nowait(self.format(record))
        except Exception:
            self.handleError(record)


def create_log_file() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir() / f"fisheye_gui_{stamp}.log"


def setup_logging(log_path: Path | None = None) -> tuple[logging.Logger, Path, Queue]:
    log_path = log_path or create_log_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    shared_processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.FILENAME]
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
        foreign_pre_chain=shared_processors,
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    log_queue: Queue = Queue()
    queue_handler = TextQueueHandler(log_queue)
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(queue_handler)

    for name in ("matplotlib", "PIL", "ultralytics"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger = logging.getLogger("fisheye_app")
    logger.info("Log file: %s", log_path)
    return logger, log_path, log_queue


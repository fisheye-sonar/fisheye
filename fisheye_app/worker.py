"""Background worker thread for FishEye GUI pipeline execution."""

from __future__ import annotations

import logging
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from fisheye_gui_pipeline.batch import BatchPlan

logger = logging.getLogger("fisheye_app")


class _StdoutToLogger(StringIO):
    def write(self, text: str) -> int:
        clean = text.strip()
        if clean:
            logger.info(clean)
        return len(text)

    def flush(self) -> None:
        pass


def _configure_compute_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


class PipelineThread(QThread):
    progress = Signal(str)
    file_progress = Signal(int, int, int, int)
    counts_progress = Signal(int, int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        base_config,
        input_paths: list[Path],
        *,
        plan: BatchPlan,
        skip_already_processed: bool = False,
    ) -> None:
        super().__init__()
        self._base_config = base_config
        self._input_paths = input_paths
        self._plan = plan
        self._skip_already_processed = skip_already_processed

    def run(self) -> None:
        _configure_compute_threads()
        self.progress.emit("Initializing pipeline…")
        self.file_progress.emit(self._plan.skip_count, 0, 0, self._plan.total)
        self.counts_progress.emit(0, 0)

        from fisheye_gui_pipeline.pipeline import run_batch_pipeline

        try:
            with redirect_stdout(_StdoutToLogger()):
                result = run_batch_pipeline(
                    self._base_config,
                    self._input_paths,
                    skip_already_processed=self._skip_already_processed,
                    plan=self._plan,
                    on_progress=self.progress.emit,
                    on_file_progress=self.file_progress.emit,
                    on_count_progress=self.counts_progress.emit,
                )
            self.finished_ok.emit(result)
        except Exception as exc:
            logger.exception("Pipeline failed")
            self.failed.emit(str(exc))

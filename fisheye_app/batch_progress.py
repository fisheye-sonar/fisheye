"""Batch file progress bar with skipped, complete, failed, and pending states."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from fisheye_gui_pipeline.batch import BatchPlan

COLOR_PENDING = QColor("#e8e8e8")
COLOR_SKIPPED = QColor("#9e9e9e")
COLOR_DONE = QColor("#0B3D91")
COLOR_FAILED = QColor("#e53935")
COLOR_BORDER = QColor("#bdbdbd")


def format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def estimate_time_remaining(
    *,
    to_process: int,
    completed: int,
    failed: int,
    run_started: float | None,
) -> str | None:
    pending = to_process - completed - failed
    if pending <= 0:
        return format_hms(0)
    if run_started is None or completed <= 0:
        return None
    elapsed = time.perf_counter() - run_started
    return format_hms((elapsed / completed) * pending)


class BatchFileProgressBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._total = 0
        self._skipped = 0
        self._completed = 0
        self._failed = 0
        self.setMinimumHeight(22)
        self.setMaximumHeight(22)

    def reset(self) -> None:
        self._total = 0
        self._skipped = 0
        self._completed = 0
        self._failed = 0
        self.update()

    def set_plan(self, plan: BatchPlan) -> None:
        self._total = plan.total
        self._skipped = plan.skip_count
        self._completed = 0
        self._failed = 0
        self.update()

    def set_progress(self, skipped: int, completed: int, failed: int, total: int) -> None:
        self._total = total
        self._skipped = skipped
        self._completed = completed
        self._failed = failed
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(COLOR_BORDER)
        painter.setBrush(COLOR_PENDING)
        painter.drawRoundedRect(rect, 4, 4)

        if self._total <= 0:
            painter.setPen(Qt.GlobalColor.black)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No files")
            return

        width = rect.width()
        x = rect.x()
        y = rect.y()
        height = rect.height()

        def segment(count: int, color: QColor, start_x: float) -> float:
            if count <= 0:
                return start_x
            seg_width = width * count / self._total
            painter.fillRect(int(start_x), y, int(seg_width), height, color)
            return start_x + seg_width

        cursor = float(x)
        cursor = segment(self._skipped, COLOR_SKIPPED, cursor)
        cursor = segment(self._failed, COLOR_FAILED, cursor)
        segment(self._completed, COLOR_DONE, cursor)

        pending = self._total - self._skipped - self._completed - self._failed
        label = (
            f"{self._skipped} skipped · {self._completed} done"
            f"{f' · {self._failed} failed' if self._failed else ''}"
            f"{f' · {pending} pending' if pending else ''}"
            f"  ({self._skipped + self._completed + self._failed}/{self._total})"
        )
        painter.setPen(Qt.GlobalColor.black)
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)


class BatchPlanPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setVisible(False)
        layout.addWidget(self._summary)

        self._bar = BatchFileProgressBar()
        self._bar.setVisible(False)
        layout.addWidget(self._bar)

        self._legend = QLabel(
            '<span style="color:#9e9e9e">■</span> already processed &nbsp; '
            '<span style="color:#0B3D91">■</span> completed this run &nbsp; '
            '<span style="color:#e53935">■</span> failed &nbsp; '
            '<span style="color:#e8e8e8">■</span> pending'
        )
        self._legend.setVisible(False)
        layout.addWidget(self._legend)

        self._eta = QLabel()
        self._eta.setVisible(False)
        layout.addWidget(self._eta)

        self._to_process = 0
        self._run_started: float | None = None

    def clear(self) -> None:
        self._summary.clear()
        self._summary.setVisible(False)
        self._bar.reset()
        self._bar.setVisible(False)
        self._legend.setVisible(False)
        self._eta.clear()
        self._eta.setVisible(False)
        self._to_process = 0
        self._run_started = None

    def show_plan(self, plan: BatchPlan, *, running: bool = False) -> None:
        if plan.total == 0:
            self.clear()
            return

        if plan.skip_count and plan.process_count:
            summary = (
                f"<b>{plan.process_count}</b> file(s) to process, "
                f"<b>{plan.skip_count}</b> already done (will skip)."
            )
        elif plan.skip_count:
            summary = (
                f"All <b>{plan.skip_count}</b> file(s) already processed — nothing to run."
            )
        else:
            summary = f"<b>{plan.process_count}</b> file(s) to process."

        if running and plan.skip_count:
            summary += " Skipped files are shown in gray on the bar."

        self._summary.setText(summary)
        self._summary.setVisible(True)
        self._bar.set_plan(plan)
        self._bar.setVisible(True)
        self._legend.setVisible(True)
        if not running:
            self._eta.setVisible(False)

    def begin_run(self, plan: BatchPlan) -> None:
        self._to_process = plan.process_count
        self._run_started = time.perf_counter() if plan.process_count else None
        self._update_eta(completed=0, failed=0)

    def _update_eta(self, *, completed: int, failed: int) -> None:
        eta = estimate_time_remaining(
            to_process=self._to_process,
            completed=completed,
            failed=failed,
            run_started=self._run_started,
        )
        self._eta.setText(f"Time remaining: {eta or '—'}")
        self._eta.setVisible(self._to_process > 0)

    def update_progress(
        self, skipped: int, completed: int, failed: int, total: int
    ) -> None:
        self._bar.set_progress(skipped, completed, failed, total)
        self._bar.setVisible(total > 0)
        self._update_eta(completed=completed, failed=failed)

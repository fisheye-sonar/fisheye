"""Application entry point."""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback
from pathlib import Path

from fisheye_app.paths import crash_log_path, user_data_dir

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _write_crash_log() -> Path:
    user_data_dir()
    path = crash_log_path()
    path.write_text(traceback.format_exc(), encoding="utf-8")
    return path


def _main() -> int:
    import matplotlib

    matplotlib.use("Agg")

    from PySide6.QtWidgets import QApplication

    from fisheye_app.logging_setup import setup_logging
    from fisheye_app.window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Fisheye - Aris salmon detection")
    app.setOrganizationName("FishEye")

    _logger, log_path, log_queue = setup_logging()
    window = MainWindow(log_queue=log_queue, log_path=log_path)
    window.show()
    return app.exec()


def main() -> int:
    if getattr(sys, "frozen", False):
        multiprocessing.freeze_support()
        exe_dir = Path(sys.executable).resolve().parent
        os.chdir(exe_dir)

    try:
        return _main()
    except Exception:
        crash_path = _write_crash_log()
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(
                None,
                "Fisheye - Aris salmon detection failed to start",
                "The application crashed on startup.\n\n"
                f"Details were written to:\n{crash_path}",
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())


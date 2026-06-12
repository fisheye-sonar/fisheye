"""Main window for the FishEye ARIS salmon detection desktop app."""

from __future__ import annotations

import logging
from pathlib import Path
from queue import Empty

from PySide6.QtCore import Qt, QSettings, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fisheye_app.batch_progress import BatchPlanPanel
from fisheye_app.paths import list_bundled_checkpoints
from fisheye_app.worker import PipelineThread
from fisheye_gui_pipeline.batch import (
    BatchPipelineResult,
    plan_batch_run,
    select_directory_files,
)
from fisheye_gui_pipeline.config import PipelineConfig
from fisheye_gui_pipeline.defaults import load_repo_defaults

SETTINGS_ORG = "FishEye"
SETTINGS_APP = "ArisSalmonDetection"

DEFAULT_CONF_THRESH = 0.10
DEFAULT_IOU_THRESH = 0.25
DEFAULT_IMAGE_SIZE = 896
DEFAULT_BATCH_SIZE = 32
DEFAULT_MIN_TARGET_LENGTH = 0.30


class MainWindow(QMainWindow):
    def __init__(self, log_queue, log_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._log_queue = log_queue
        self._log_path = log_path
        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._repo_defaults = load_repo_defaults()
        self._pipeline_thread: PipelineThread | None = None
        self._last_output_dir: Path | None = None
        self._input_paths: list[Path] = []
        self._source_directory: Path | None = None

        self.setWindowTitle("Fisheye - Aris salmon detection")
        self.resize(720, 640)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        files_box = QGroupBox("Inputs")
        files_form = QFormLayout(files_box)

        self._inputs_edit = QLineEdit()
        self._inputs_edit.setReadOnly(True)
        self._inputs_edit.setPlaceholderText("No files selected")
        self._add_files_btn = QPushButton("Add files…")
        self._add_files_btn.clicked.connect(self._browse_input_files)
        self._add_dir_btn = QPushButton("Add directory…")
        self._add_dir_btn.clicked.connect(self._browse_input_directory)
        self._clear_inputs_btn = QPushButton("Clear")
        self._clear_inputs_btn.clicked.connect(self._clear_input_selection)
        files_row = QHBoxLayout()
        files_row.addWidget(self._inputs_edit, stretch=1)
        files_row.addWidget(self._add_files_btn)
        files_row.addWidget(self._add_dir_btn)
        files_row.addWidget(self._clear_inputs_btn)
        files_form.addRow("ARIS input:", files_row)

        dir_opts = QWidget()
        dir_opts_layout = QVBoxLayout(dir_opts)
        dir_opts_layout.setContentsMargins(0, 0, 0, 0)
        self._dir_mode_all = QRadioButton("All files in directory (A–Z)")
        self._dir_mode_limit = QRadioButton("First N files in directory (A–Z)")
        self._dir_mode_all.setChecked(True)
        self._dir_mode_group = QButtonGroup(self)
        self._dir_mode_group.addButton(self._dir_mode_all)
        self._dir_mode_group.addButton(self._dir_mode_limit)
        dir_opts_layout.addWidget(self._dir_mode_all)
        limit_row = QHBoxLayout()
        limit_row.addWidget(self._dir_mode_limit)
        self._dir_limit_n = QSpinBox()
        self._dir_limit_n.setRange(1, 1_000_000)
        self._dir_limit_n.setValue(10)
        self._dir_limit_n.setEnabled(False)
        limit_row.addWidget(self._dir_limit_n)
        limit_row.addStretch()
        dir_opts_layout.addLayout(limit_row)
        self._dir_options = QGroupBox("Directory selection")
        self._dir_options.setLayout(dir_opts_layout)
        self._dir_options.setEnabled(False)
        files_form.addRow(self._dir_options)

        self._skip_processed = QCheckBox("Skip already processed")
        self._skip_processed.setChecked(True)
        files_form.addRow(self._skip_processed)

        self._dir_mode_all.toggled.connect(self._on_dir_mode_changed)
        self._dir_mode_limit.toggled.connect(self._on_dir_mode_changed)
        self._dir_limit_n.valueChanged.connect(self._refresh_directory_selection)
        self._skip_processed.toggled.connect(self._refresh_batch_preview)

        self._output_edit = QLineEdit()
        self._output_btn = QPushButton("Browse…")
        self._output_btn.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_edit)
        output_row.addWidget(self._output_btn)
        files_form.addRow("Output directory:", output_row)

        layout.addWidget(files_box)

        model_box = QGroupBox("Model")
        model_form = QFormLayout(model_box)

        self._checkpoint_combo = QComboBox()
        self._checkpoint_combo.currentIndexChanged.connect(
            self._on_checkpoint_combo_changed
        )
        model_form.addRow("Checkpoints:", self._checkpoint_combo)

        self._use_other_checkpoint = QCheckBox("Use other checkpoint")
        self._use_other_checkpoint.toggled.connect(self._on_use_other_checkpoint)
        model_form.addRow("", self._use_other_checkpoint)

        self._checkpoint_edit = QLineEdit()
        self._checkpoint_edit.setPlaceholderText("Browse for a .pt file…")
        self._checkpoint_btn = QPushButton("Browse…")
        self._checkpoint_btn.clicked.connect(self._browse_checkpoint)
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(self._checkpoint_edit)
        ckpt_row.addWidget(self._checkpoint_btn)
        model_form.addRow("Other checkpoint:", ckpt_row)

        self._refresh_bundled_checkpoints()
        self._on_use_other_checkpoint(False)
        layout.addWidget(model_box)

        opts_box = QGroupBox("Options")
        opts_form = QFormLayout(opts_box)

        upstream_row = QWidget()
        upstream_layout = QHBoxLayout(upstream_row)
        upstream_layout.setContentsMargins(0, 0, 0, 0)
        self._upstream_left = QRadioButton("Left")
        self._upstream_right = QRadioButton("Right")
        self._upstream_left.setChecked(True)
        self._upstream_group = QButtonGroup(self)
        self._upstream_group.addButton(self._upstream_left)
        self._upstream_group.addButton(self._upstream_right)
        upstream_layout.addWidget(self._upstream_left)
        upstream_layout.addWidget(self._upstream_right)
        upstream_layout.addStretch()
        opts_form.addRow("Upstream:", upstream_row)

        self._start_frame = QSpinBox()
        self._start_frame.setRange(0, 10_000_000)
        self._start_frame.setValue(0)

        self._end_frame = QSpinBox()
        self._end_frame.setRange(-1, 10_000_000)
        self._end_frame.setValue(-1)
        self._end_frame.setSpecialValueText("end of file")
        self._end_frame.setToolTip("Exclusive end frame index; -1 means end of file")

        self._analyze_all = QCheckBox("Analyse all frames")
        self._analyze_all.setChecked(True)
        self._analyze_all.setToolTip(
            "When checked, process all frames (start=0, end=end of file)"
        )

        frames_row = QWidget()
        frames_layout = QHBoxLayout(frames_row)
        frames_layout.setContentsMargins(0, 0, 0, 0)
        frames_layout.addWidget(QLabel("Start"))
        frames_layout.addWidget(self._start_frame)
        frames_layout.addSpacing(8)
        frames_layout.addWidget(QLabel("End"))
        frames_layout.addWidget(self._end_frame)
        frames_layout.addStretch()

        frames_container = QWidget()
        frames_container_layout = QHBoxLayout(frames_container)
        frames_container_layout.setContentsMargins(0, 0, 0, 0)
        frames_container_layout.addWidget(self._analyze_all)
        frames_container_layout.addWidget(frames_row, stretch=1)
        opts_form.addRow(frames_container)

        self._export_summary_csv = QCheckBox("Summary CSV")
        self._export_summary_csv.setChecked(True)
        self._export_detailed_csv = QCheckBox("Detailed CSV")
        self._export_detailed_csv.setChecked(True)
        self._export_fc = QCheckBox("FC")
        self._export_fc.setChecked(True)
        self._export_xml = QCheckBox("XML")
        self._export_xml.setChecked(True)
        self._export_mot = QCheckBox("MOT")
        self._export_mot.setChecked(False)
        exports_row = QWidget()
        exports_layout = QHBoxLayout(exports_row)
        exports_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (
            self._export_summary_csv,
            self._export_detailed_csv,
            self._export_fc,
            self._export_xml,
            self._export_mot,
        ):
            exports_layout.addWidget(widget)
        exports_layout.addStretch()
        opts_form.addRow(exports_row)

        layout.addWidget(opts_box)

        self._adv_toggle = QToolButton()
        self._adv_toggle.setText("Advanced")
        self._adv_toggle.setCheckable(True)
        self._adv_toggle.setChecked(False)
        self._adv_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._adv_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._adv_toggle.toggled.connect(self._on_advanced_toggled)

        adv_header_row = QHBoxLayout()
        adv_header_row.addWidget(self._adv_toggle)
        adv_header_row.addStretch()
        self._adv_reset_btn = QPushButton("Reset all")
        self._adv_reset_btn.clicked.connect(self._reset_all_options)
        adv_header_row.addWidget(self._adv_reset_btn)
        layout.addLayout(adv_header_row)

        self._adv_content = QWidget()
        adv_layout = QHBoxLayout(self._adv_content)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(24)

        adv_left = QWidget()
        adv_left_form = QFormLayout(adv_left)
        adv_right = QWidget()
        adv_right_form = QFormLayout(adv_right)
        adv_layout.addWidget(adv_left, stretch=1)
        adv_layout.addWidget(adv_right, stretch=1)

        self._device = QComboBox()
        self._device.addItems(self._available_devices())
        adv_left_form.addRow("Device:", self._device)

        self._detector_type = QComboBox()
        self._detector_type.addItems(["yolov5", "yolov11", "yolov26"])
        adv_left_form.addRow("Detector type:", self._detector_type)

        self._conf_thresh = QDoubleSpinBox()
        self._conf_thresh.setRange(0.0, 1.0)
        self._conf_thresh.setSingleStep(0.01)
        self._conf_thresh.setDecimals(2)
        self._conf_thresh.setValue(self._repo_defaults.conf)
        adv_left_form.addRow("Confidence:", self._conf_thresh)

        self._iou_thresh = QDoubleSpinBox()
        self._iou_thresh.setRange(0.0, 1.0)
        self._iou_thresh.setSingleStep(0.01)
        self._iou_thresh.setDecimals(2)
        self._iou_thresh.setValue(self._repo_defaults.iou)
        adv_left_form.addRow("IoU:", self._iou_thresh)

        self._image_size = QSpinBox()
        self._image_size.setRange(32, 4096)
        self._image_size.setSingleStep(32)
        self._image_size.setValue(self._repo_defaults.image_size)
        adv_left_form.addRow("Image size:", self._image_size)

        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 256)
        self._batch_size.setValue(self._repo_defaults.batch_size)
        adv_left_form.addRow("Batch size:", self._batch_size)

        self._workers = QSpinBox()
        self._workers.setRange(0, 64)
        self._workers.setValue(self._repo_defaults.workers)
        adv_left_form.addRow("Data workers:", self._workers)

        self._max_workers = QSpinBox()
        self._max_workers.setRange(1, 64)
        self._max_workers.setValue(self._repo_defaults.inference_max_workers)
        adv_left_form.addRow("Inference workers:", self._max_workers)

        self._min_target_length = QDoubleSpinBox()
        self._min_target_length.setRange(0.0, 20.0)
        self._min_target_length.setSingleStep(0.05)
        self._min_target_length.setDecimals(2)
        self._min_target_length.setValue(self._repo_defaults.min_target_length)
        adv_right_form.addRow("Min target (m):", self._min_target_length)

        self._max_target_length = QDoubleSpinBox()
        self._max_target_length.setRange(0.0, 20.0)
        self._max_target_length.setSingleStep(0.05)
        self._max_target_length.setDecimals(2)
        self._max_target_length.setValue(self._repo_defaults.max_target_length)
        self._max_target_length.setSpecialValueText("disabled")
        adv_right_form.addRow("Max target (m):", self._max_target_length)

        self._distance_offset = QDoubleSpinBox()
        self._distance_offset.setRange(-50.0, 50.0)
        self._distance_offset.setSingleStep(0.1)
        self._distance_offset.setDecimals(2)
        self._distance_offset.setValue(self._repo_defaults.distance_offset)
        adv_right_form.addRow("Distance offset:", self._distance_offset)

        self._tracker_max_age = QSpinBox()
        self._tracker_max_age.setRange(0, 10_000)
        self._tracker_max_age.setValue(self._repo_defaults.tracker_max_age)
        adv_right_form.addRow("Tracker max age:", self._tracker_max_age)

        self._tracker_min_hits = QSpinBox()
        self._tracker_min_hits.setRange(0, 10_000)
        self._tracker_min_hits.setValue(self._repo_defaults.tracker_min_hits)
        adv_right_form.addRow("Tracker min hits:", self._tracker_min_hits)

        self._tracker_min_travel = QSpinBox()
        self._tracker_min_travel.setRange(0, 10_000)
        self._tracker_min_travel.setValue(self._repo_defaults.tracker_min_travel)
        adv_right_form.addRow("Tracker min travel:", self._tracker_min_travel)

        self._tracker_iou_threshold = QDoubleSpinBox()
        self._tracker_iou_threshold.setRange(0.0, 1.0)
        self._tracker_iou_threshold.setDecimals(4)
        self._tracker_iou_threshold.setSingleStep(0.001)
        self._tracker_iou_threshold.setValue(self._repo_defaults.tracker_iou_threshold)
        adv_right_form.addRow("Tracker IoU:", self._tracker_iou_threshold)

        self._tracker_reverse = QCheckBox("Reverse tracking order")
        self._tracker_reverse.setChecked(self._repo_defaults.tracker_reverse)
        adv_right_form.addRow(self._tracker_reverse)

        self._use_blur = QCheckBox("Blur background subtraction input")
        self._use_blur.setChecked(self._repo_defaults.use_blur)
        adv_right_form.addRow(self._use_blur)

        self._use_multithreading = QCheckBox("Parallelize inference")
        self._use_multithreading.setChecked(
            self._repo_defaults.inference_use_multithreading
        )
        adv_right_form.addRow(self._use_multithreading)

        self._adv_content.setVisible(False)
        layout.addWidget(self._adv_content)

        button_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #0B3D91;
                color: white;
            }
            QPushButton:hover {
                background-color: #1550B8;
            }
            QPushButton:pressed {
                background-color: #082E6B;
            }
            QPushButton:disabled {
                background-color: #7F9DCD;
                color: white;
            }
            """
        )
        self._run_btn.clicked.connect(self._on_run)
        self._open_out_btn = QPushButton("Open output folder")
        self._open_out_btn.clicked.connect(self._open_output_folder)
        self._open_out_btn.setEnabled(False)
        self._open_log_btn = QPushButton("Open log file")
        self._open_log_btn.clicked.connect(self._open_log_file)
        button_row.addWidget(self._run_btn)
        button_row.addWidget(self._open_out_btn)
        button_row.addStretch()
        button_row.addWidget(self._open_log_btn)
        layout.addLayout(button_row)

        self._status = QLabel("Ready.")
        layout.addWidget(self._status)

        self._run_counts = QLabel()
        self._set_run_counts(0, 0)
        layout.addWidget(self._run_counts)

        self._batch_plan_panel = BatchPlanPanel()
        layout.addWidget(self._batch_plan_panel)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setPlaceholderText("Log output…")
        layout.addWidget(self._log_view, stretch=1)

        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._drain_log_queue)
        self._log_timer.start(200)

        def _on_analyze_all_toggled(checked: bool) -> None:
            self._start_frame.setEnabled(not checked)
            self._end_frame.setEnabled(not checked)
            if checked:
                self._start_frame.setValue(0)
                self._end_frame.setValue(-1)
            self._refresh_batch_preview()

        self._analyze_all.toggled.connect(_on_analyze_all_toggled)
        _on_analyze_all_toggled(self._analyze_all.isChecked())

        for widget in (
            self._output_edit,
            self._export_summary_csv,
            self._export_detailed_csv,
            self._export_fc,
            self._export_xml,
            self._export_mot,
            self._conf_thresh,
            self._iou_thresh,
            self._image_size,
            self._batch_size,
            self._workers,
            self._max_workers,
            self._min_target_length,
            self._max_target_length,
            self._distance_offset,
            self._tracker_max_age,
            self._tracker_min_hits,
            self._tracker_min_travel,
            self._tracker_iou_threshold,
            self._use_blur,
            self._use_multithreading,
            self._detector_type,
            self._device,
        ):
            signal = getattr(widget, "textChanged", None)
            if signal is not None:
                signal.connect(self._refresh_batch_preview)
        self._output_edit.textChanged.connect(self._refresh_batch_preview)
        self._start_frame.valueChanged.connect(self._refresh_batch_preview)
        self._end_frame.valueChanged.connect(self._refresh_batch_preview)
        self._export_summary_csv.toggled.connect(self._refresh_batch_preview)
        self._export_detailed_csv.toggled.connect(self._refresh_batch_preview)
        self._export_fc.toggled.connect(self._refresh_batch_preview)
        self._export_xml.toggled.connect(self._refresh_batch_preview)
        self._export_mot.toggled.connect(self._refresh_batch_preview)
        self._conf_thresh.valueChanged.connect(self._refresh_batch_preview)
        self._iou_thresh.valueChanged.connect(self._refresh_batch_preview)
        self._image_size.valueChanged.connect(self._refresh_batch_preview)
        self._batch_size.valueChanged.connect(self._refresh_batch_preview)
        self._workers.valueChanged.connect(self._refresh_batch_preview)
        self._max_workers.valueChanged.connect(self._refresh_batch_preview)
        self._min_target_length.valueChanged.connect(self._refresh_batch_preview)
        self._max_target_length.valueChanged.connect(self._refresh_batch_preview)
        self._distance_offset.valueChanged.connect(self._refresh_batch_preview)
        self._tracker_max_age.valueChanged.connect(self._refresh_batch_preview)
        self._tracker_min_hits.valueChanged.connect(self._refresh_batch_preview)
        self._tracker_min_travel.valueChanged.connect(self._refresh_batch_preview)
        self._tracker_iou_threshold.valueChanged.connect(self._refresh_batch_preview)
        self._tracker_reverse.toggled.connect(self._refresh_batch_preview)
        self._use_blur.toggled.connect(self._refresh_batch_preview)
        self._use_multithreading.toggled.connect(self._refresh_batch_preview)
        self._detector_type.currentIndexChanged.connect(self._refresh_batch_preview)
        self._device.currentIndexChanged.connect(self._refresh_batch_preview)

        self._load_settings()
        self._update_input_summary()
        self._refresh_batch_preview()

    def _on_advanced_toggled(self, expanded: bool) -> None:
        self._adv_content.setVisible(expanded)
        self._adv_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def _is_pipeline_running(self) -> bool:
        return self._pipeline_thread is not None and self._pipeline_thread.isRunning()

    @staticmethod
    def _available_devices() -> list[str]:
        devices = ["cpu"]
        try:
            import torch

            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    devices.append(f"cuda:{index}")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                devices.append("mps")
        except ImportError:
            pass
        return devices

    def _refresh_bundled_checkpoints(self) -> None:
        self._checkpoint_combo.blockSignals(True)
        self._checkpoint_combo.clear()
        self._checkpoint_combo.addItem("(none)", "")
        for path in list_bundled_checkpoints():
            self._checkpoint_combo.addItem(path.name, str(path))
        self._checkpoint_combo.blockSignals(False)
        if not self._use_other_checkpoint.isChecked():
            self._select_default_bundled_checkpoint()

    def _select_default_bundled_checkpoint(self) -> None:
        target = str(self._repo_defaults.checkpoint)
        idx = self._checkpoint_combo.findData(target)
        if idx >= 0:
            self._checkpoint_combo.setCurrentIndex(idx)
        elif self._checkpoint_combo.count() > 1:
            self._checkpoint_combo.setCurrentIndex(1)

    def _on_checkpoint_combo_changed(self, index: int) -> None:
        if self._use_other_checkpoint.isChecked():
            return
        path = self._checkpoint_combo.itemData(index)
        if path:
            self._checkpoint_edit.setText(path)
        self._refresh_batch_preview()

    def _on_use_other_checkpoint(self, checked: bool) -> None:
        self._update_checkpoint_controls()
        if checked and not self._checkpoint_edit.text().strip():
            path = self._checkpoint_combo.currentData()
            if path:
                self._checkpoint_edit.setText(path)
        elif not checked:
            self._on_checkpoint_combo_changed(self._checkpoint_combo.currentIndex())

    def _update_checkpoint_controls(self) -> None:
        running = self._is_pipeline_running()
        use_other = self._use_other_checkpoint.isChecked()
        self._checkpoint_combo.setEnabled(not running and not use_other)
        self._use_other_checkpoint.setEnabled(not running)
        self._checkpoint_edit.setEnabled(not running and use_other)
        self._checkpoint_btn.setEnabled(not running and use_other)

    def _checkpoint_path(self) -> Path:
        if self._use_other_checkpoint.isChecked():
            return Path(self._checkpoint_edit.text().strip())
        path = self._checkpoint_combo.currentData()
        if path:
            return Path(path)
        return Path(self._checkpoint_edit.text().strip())

    def _browse_checkpoint(self) -> None:
        start = self._checkpoint_edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select checkpoint",
            start,
            "PyTorch weights (*.pt);;All files (*)",
        )
        if path:
            self._checkpoint_edit.setText(path)
            self._refresh_batch_preview()

    def _browse_input_files(self) -> None:
        start_dir = str(self._source_directory or Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select ARIS files",
            start_dir,
            "Sonar files (*.aris *.ddf);;All files (*)",
        )
        if not paths:
            return
        self._source_directory = None
        self._dir_options.setEnabled(False)
        existing = {path.resolve() for path in self._input_paths}
        for path_text in paths:
            path = Path(path_text).resolve()
            if path not in existing:
                self._input_paths.append(path)
                existing.add(path)
        self._input_paths.sort(key=lambda path: path.name.lower())
        self._update_input_summary()
        self._maybe_default_output_dir()
        self._refresh_batch_preview()

    def _browse_input_directory(self) -> None:
        start_dir = str(self._source_directory or Path.home())
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select directory containing ARIS files",
            start_dir,
        )
        if not dir_path:
            return
        self._source_directory = Path(dir_path)
        self._dir_options.setEnabled(True)
        self._refresh_directory_selection()
        self._maybe_default_output_dir()
        self._refresh_batch_preview()

    def _clear_input_selection(self) -> None:
        self._input_paths.clear()
        self._source_directory = None
        self._dir_options.setEnabled(False)
        self._update_input_summary()
        self._refresh_batch_preview()

    def _on_dir_mode_changed(self) -> None:
        self._dir_limit_n.setEnabled(self._dir_mode_limit.isChecked())
        self._refresh_directory_selection()

    def _refresh_directory_selection(self) -> None:
        if self._source_directory is None:
            return
        try:
            self._input_paths = select_directory_files(
                self._source_directory,
                run_all=self._dir_mode_all.isChecked(),
                limit=self._dir_limit_n.value(),
            )
        except (NotADirectoryError, OSError) as exc:
            QMessageBox.warning(self, "Directory error", str(exc))
            return
        if not self._input_paths:
            QMessageBox.warning(
                self,
                "No files found",
                f"No .aris or .ddf files found in:\n{self._source_directory}",
            )
        self._update_input_summary()
        self._refresh_batch_preview()

    def _try_build_plan(self):
        if not self._input_paths:
            return None
        try:
            config = self._build_base_config()
            return plan_batch_run(
                config,
                self._input_paths,
                skip_already_processed=self._skip_processed.isChecked(),
            )
        except (ValueError, FileNotFoundError):
            return None

    def _refresh_batch_preview(self) -> None:
        if not hasattr(self, "_batch_plan_panel"):
            return
        if self._is_pipeline_running():
            return
        plan = self._try_build_plan()
        if plan is None:
            self._batch_plan_panel.clear()
            return
        self._batch_plan_panel.show_plan(plan)

    def _update_input_summary(self) -> None:
        if not self._input_paths:
            self._inputs_edit.setText("")
            return
        if self._source_directory is not None:
            mode = (
                "all"
                if self._dir_mode_all.isChecked()
                else f"first {self._dir_limit_n.value()}"
            )
            self._inputs_edit.setText(
                f"{self._source_directory} — {len(self._input_paths)} file(s), {mode}"
            )
        elif len(self._input_paths) == 1:
            self._inputs_edit.setText(str(self._input_paths[0]))
        else:
            self._inputs_edit.setText(f"{len(self._input_paths)} files selected")

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            self._output_edit.text() or str(Path.home()),
        )
        if path:
            self._output_edit.setText(path)

    def _maybe_default_output_dir(self) -> None:
        if self._output_edit.text().strip():
            return
        if self._source_directory is not None:
            self._output_edit.setText(str(self._source_directory / "outputs"))
        elif self._input_paths:
            self._output_edit.setText(str(self._input_paths[0].parent / "outputs"))

    def _upstream_direction(self) -> str:
        return "right" if self._upstream_right.isChecked() else "left"

    def _default_device_index(self) -> int:
        idx = self._device.findText("cpu")
        return idx if idx >= 0 else 0

    def _reset_all_options(self) -> None:
        self._settings.clear()
        self._use_other_checkpoint.setChecked(False)
        self._refresh_bundled_checkpoints()
        self._output_edit.clear()
        self._skip_processed.setChecked(True)
        self._analyze_all.setChecked(True)
        self._start_frame.setValue(0)
        self._end_frame.setValue(-1)
        if self._repo_defaults.upstream_direction == "right":
            self._upstream_right.setChecked(True)
        else:
            self._upstream_left.setChecked(True)
        self._export_summary_csv.setChecked(True)
        self._export_detailed_csv.setChecked(True)
        self._export_fc.setChecked(True)
        self._export_xml.setChecked(True)
        self._export_mot.setChecked(False)
        self._device.setCurrentText(self._repo_defaults.device)
        if self._device.currentText() != self._repo_defaults.device:
            self._device.setCurrentIndex(self._default_device_index())
        self._detector_type.setCurrentText(self._repo_defaults.detector_type)
        self._conf_thresh.setValue(self._repo_defaults.conf)
        self._iou_thresh.setValue(self._repo_defaults.iou)
        self._image_size.setValue(self._repo_defaults.image_size)
        self._batch_size.setValue(self._repo_defaults.batch_size)
        self._workers.setValue(self._repo_defaults.workers)
        self._max_workers.setValue(self._repo_defaults.inference_max_workers)
        self._min_target_length.setValue(self._repo_defaults.min_target_length)
        self._max_target_length.setValue(self._repo_defaults.max_target_length)
        self._distance_offset.setValue(self._repo_defaults.distance_offset)
        self._tracker_max_age.setValue(self._repo_defaults.tracker_max_age)
        self._tracker_min_hits.setValue(self._repo_defaults.tracker_min_hits)
        self._tracker_min_travel.setValue(self._repo_defaults.tracker_min_travel)
        self._tracker_iou_threshold.setValue(
            self._repo_defaults.tracker_iou_threshold
        )
        self._tracker_reverse.setChecked(self._repo_defaults.tracker_reverse)
        self._use_blur.setChecked(self._repo_defaults.use_blur)
        self._use_multithreading.setChecked(
            self._repo_defaults.inference_use_multithreading
        )
        self._maybe_default_output_dir()
        self._save_settings()
        self._refresh_batch_preview()

    def _selected_export_options(self) -> list[str]:
        options: list[str] = []
        if self._export_summary_csv.isChecked():
            options.append("summary_csv")
        if self._export_detailed_csv.isChecked():
            options.append("detailed_csv")
        if self._export_fc.isChecked():
            options.append("fc")
        if self._export_xml.isChecked():
            options.append("xml")
        if self._export_mot.isChecked():
            options.append("mot")
        return options

    def _build_base_config(self) -> PipelineConfig:
        output_text = self._output_edit.text().strip()
        placeholder = self._input_paths[0] if self._input_paths else Path(".")
        return PipelineConfig(
            input_path=placeholder,
            checkpoint=self._checkpoint_path(),
            start_frame=self._start_frame.value(),
            end_frame=self._end_frame.value(),
            output_dir=Path(output_text) if output_text else None,
            device=self._device.currentText(),
            detector_type=self._detector_type.currentText(),
            batch_size=self._batch_size.value(),
            workers=self._workers.value(),
            image_size=self._image_size.value(),
            use_blur=self._use_blur.isChecked(),
            dataset_use_multithreading=self._repo_defaults.dataset_use_multithreading,
            dataset_max_workers=self._repo_defaults.dataset_max_workers,
            use_multithreading=self._use_multithreading.isChecked(),
            max_workers=self._max_workers.value(),
            conf=self._conf_thresh.value(),
            iou=self._iou_thresh.value(),
            upstream_direction=self._upstream_direction(),
            distance_offset=self._distance_offset.value(),
            min_target_length=self._min_target_length.value(),
            max_target_length=self._max_target_length.value(),
            tracker_max_age=self._tracker_max_age.value(),
            tracker_min_hits=self._tracker_min_hits.value(),
            tracker_min_travel=self._tracker_min_travel.value(),
            tracker_iou_threshold=self._tracker_iou_threshold.value(),
            tracker_reverse=self._tracker_reverse.isChecked(),
            export_options=self._selected_export_options(),
        )

    def _validate_inputs(self) -> tuple[PipelineConfig, list[Path]]:
        if not self._input_paths:
            raise ValueError("Select at least one ARIS file")
        config = self._build_base_config()
        config.validate_shared()
        if config.output_dir is None and len(self._input_paths) > 1:
            raise ValueError("Set an output directory when processing multiple files")
        for path in self._input_paths:
            if path.suffix.lower() not in (".aris", ".ddf"):
                raise ValueError(f"Expected .aris or .ddf file: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"ARIS file not found: {path}")
        return config, self._input_paths

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("Running…" if running else "Run")
        for widget in (
            self._inputs_edit,
            self._add_files_btn,
            self._add_dir_btn,
            self._clear_inputs_btn,
            self._dir_options,
            self._dir_mode_all,
            self._dir_mode_limit,
            self._dir_limit_n,
            self._skip_processed,
            self._output_edit,
            self._output_btn,
            self._start_frame,
            self._end_frame,
            self._analyze_all,
            self._upstream_left,
            self._upstream_right,
            self._export_summary_csv,
            self._export_detailed_csv,
            self._export_fc,
            self._export_xml,
            self._export_mot,
            self._adv_toggle,
            self._adv_reset_btn,
            self._device,
            self._detector_type,
            self._conf_thresh,
            self._iou_thresh,
            self._image_size,
            self._batch_size,
            self._workers,
            self._max_workers,
            self._min_target_length,
            self._max_target_length,
            self._distance_offset,
            self._tracker_max_age,
            self._tracker_min_hits,
            self._tracker_min_travel,
            self._tracker_iou_threshold,
            self._tracker_reverse,
            self._use_blur,
            self._use_multithreading,
        ):
            widget.setEnabled(not running)
        self._update_checkpoint_controls()

    def _on_run(self) -> None:
        if self._is_pipeline_running():
            return
        try:
            config, input_paths = self._validate_inputs()
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return

        plan = plan_batch_run(
            config,
            input_paths,
            skip_already_processed=self._skip_processed.isChecked(),
        )
        self._batch_plan_panel.show_plan(plan, running=True)
        self._batch_plan_panel.begin_run(plan)
        self._batch_plan_panel.update_progress(plan.skip_count, 0, 0, plan.total)

        if plan.process_count == 0:
            QMessageBox.information(
                self,
                "Nothing to run",
                f"All {plan.skip_count} selected file(s) are already processed.",
            )
            self._status.setText(f"All {plan.skip_count} file(s) already processed.")
            return

        self._save_settings()
        self._set_running(True)
        logging.getLogger("fisheye_app").info(
            "Starting pipeline (%d to process, %d skipped)",
            plan.process_count,
            plan.skip_count,
        )

        thread = PipelineThread(
            config,
            input_paths,
            plan=plan,
            skip_already_processed=self._skip_processed.isChecked(),
        )
        thread.progress.connect(self._on_progress)
        thread.file_progress.connect(self._on_file_progress)
        thread.counts_progress.connect(self._on_counts_progress)
        thread.finished_ok.connect(self._on_finished)
        thread.failed.connect(self._on_failed)
        thread.finished.connect(self._on_thread_finished)
        self._pipeline_thread = thread
        thread.start()

    def _on_thread_finished(self) -> None:
        self._pipeline_thread = None

    def _on_file_progress(
        self, skipped: int, completed: int, failed: int, total: int
    ) -> None:
        self._batch_plan_panel.update_progress(skipped, completed, failed, total)

    def _on_progress(self, message: str) -> None:
        self._status.setText(message)

    def _set_run_counts(self, upstream: int, downstream: int) -> None:
        self._run_counts.setText(
            f"Total upstream: {upstream}    Total downstream: {downstream}"
        )

    def _on_counts_progress(self, upstream: int, downstream: int) -> None:
        self._set_run_counts(upstream, downstream)

    def _on_finished(self, result: BatchPipelineResult) -> None:
        self._set_running(False)
        self._refresh_batch_preview()
        self._set_run_counts(result.total_upstream, result.total_downstream)
        if result.plan is not None:
            self._batch_plan_panel.update_progress(
                result.plan.skip_count,
                result.processed_count,
                result.failed_count,
                result.plan.total,
            )
        self._last_output_dir = result.output_dir
        self._open_out_btn.setEnabled(True)
        self._status.setText(
            f"Done: {result.processed_count} processed, "
            f"{result.skipped_count} skipped, {result.failed_count} failed "
            f"({result.total_seconds:.1f}s)"
        )

        lines = [
            f"Processed: {result.processed_count}",
            f"Skipped: {result.skipped_count}",
            f"Failed: {result.failed_count}",
            f"Upstream: {result.total_upstream}",
            f"Downstream: {result.total_downstream}",
            f"Output: {result.output_dir}",
        ]
        if result.results:
            last = result.results[-1]
            if last.exported_paths:
                lines.extend(["", "Last output:", str(last.exported_paths[0])])
        if result.failures:
            lines.append("")
            lines.append("Failures:")
            for path, message in result.failures[:10]:
                lines.append(f"  {path.name}: {message}")
            if len(result.failures) > 10:
                lines.append(f"  … and {len(result.failures) - 10} more")

        title = "Complete" if len(self._input_paths) == 1 else "Batch complete"
        QMessageBox.information(self, title, "\n".join(lines))

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._refresh_batch_preview()
        self._status.setText("Failed.")
        logging.getLogger("fisheye_app").error("Pipeline failed: %s", message)
        QMessageBox.critical(self, "Pipeline failed", message)

    def _drain_log_queue(self) -> None:
        while True:
            try:
                message = self._log_queue.get_nowait()
            except Empty:
                break
            self._log_view.append(message)

    def _open_output_folder(self) -> None:
        if self._last_output_dir and self._last_output_dir.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output_dir)))

    def _open_log_file(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._log_path)))

    def _load_settings(self) -> None:
        saved_checkpoint = self._settings.value("checkpoint", "", type=str)
        use_other = self._settings.value("use_other_checkpoint", False, type=bool)
        self._use_other_checkpoint.setChecked(use_other)
        if use_other:
            self._checkpoint_edit.setText(saved_checkpoint)
        elif saved_checkpoint:
            index = self._checkpoint_combo.findData(saved_checkpoint)
            if index >= 0:
                self._checkpoint_combo.setCurrentIndex(index)

        self._output_edit.setText(self._settings.value("output_dir", "", type=str))
        self._skip_processed.setChecked(
            self._settings.value("skip_processed", True, type=bool)
        )
        self._analyze_all.setChecked(
            self._settings.value("analyze_all_frames", True, type=bool)
        )
        self._start_frame.setValue(self._settings.value("start_frame", 0, type=int))
        self._end_frame.setValue(self._settings.value("end_frame", -1, type=int))
        self._upstream_right.setChecked(
            self._settings.value(
                "upstream_direction",
                self._repo_defaults.upstream_direction,
                type=str,
            )
            == "right"
        )
        self._device.setCurrentText(
            self._settings.value("device", self._repo_defaults.device, type=str)
        )
        self._detector_type.setCurrentText(
            self._settings.value(
                "detector_type", self._repo_defaults.detector_type, type=str
            )
        )
        self._conf_thresh.setValue(
            self._settings.value("conf_thresh", self._repo_defaults.conf, type=float)
        )
        self._iou_thresh.setValue(
            self._settings.value("iou_thresh", self._repo_defaults.iou, type=float)
        )
        self._image_size.setValue(
            self._settings.value(
                "image_size", self._repo_defaults.image_size, type=int
            )
        )
        self._batch_size.setValue(
            self._settings.value(
                "batch_size", self._repo_defaults.batch_size, type=int
            )
        )
        self._workers.setValue(
            self._settings.value("workers", self._repo_defaults.workers, type=int)
        )
        self._max_workers.setValue(
            self._settings.value(
                "max_workers", self._repo_defaults.inference_max_workers, type=int
            )
        )
        self._tracker_max_age.setValue(
            self._settings.value(
                "tracker_max_age", self._repo_defaults.tracker_max_age, type=int
            )
        )
        self._tracker_min_hits.setValue(
            self._settings.value(
                "tracker_min_hits", self._repo_defaults.tracker_min_hits, type=int
            )
        )
        self._tracker_min_travel.setValue(
            self._settings.value(
                "tracker_min_travel",
                self._repo_defaults.tracker_min_travel,
                type=int,
            )
        )
        self._tracker_iou_threshold.setValue(
            self._settings.value(
                "tracker_iou_threshold",
                self._repo_defaults.tracker_iou_threshold,
                type=float,
            )
        )
        self._tracker_reverse.setChecked(
            self._settings.value(
                "tracker_reverse", self._repo_defaults.tracker_reverse, type=bool
            )
        )
        self._min_target_length.setValue(
            self._settings.value(
                "min_target_length", self._repo_defaults.min_target_length, type=float
            )
        )
        self._max_target_length.setValue(
            self._settings.value(
                "max_target_length", self._repo_defaults.max_target_length, type=float
            )
        )
        self._distance_offset.setValue(
            self._settings.value(
                "distance_offset", self._repo_defaults.distance_offset, type=float
            )
        )
        self._use_blur.setChecked(
            self._settings.value("use_blur", self._repo_defaults.use_blur, type=bool)
        )
        self._use_multithreading.setChecked(
            self._settings.value(
                "use_multithreading",
                self._repo_defaults.inference_use_multithreading,
                type=bool,
            )
        )
        self._export_summary_csv.setChecked(
            self._settings.value("export_summary_csv", True, type=bool)
        )
        self._export_detailed_csv.setChecked(
            self._settings.value("export_detailed_csv", True, type=bool)
        )
        self._export_fc.setChecked(
            self._settings.value("export_fc", True, type=bool)
        )
        self._export_xml.setChecked(
            self._settings.value("export_xml", True, type=bool)
        )
        self._export_mot.setChecked(
            self._settings.value("export_mot", False, type=bool)
        )

    def _save_settings(self) -> None:
        self._settings.setValue("checkpoint", str(self._checkpoint_path()))
        self._settings.setValue(
            "use_other_checkpoint", self._use_other_checkpoint.isChecked()
        )
        self._settings.setValue("output_dir", self._output_edit.text().strip())
        self._settings.setValue("skip_processed", self._skip_processed.isChecked())
        self._settings.setValue("analyze_all_frames", self._analyze_all.isChecked())
        self._settings.setValue("start_frame", self._start_frame.value())
        self._settings.setValue("end_frame", self._end_frame.value())
        self._settings.setValue("upstream_direction", self._upstream_direction())
        self._settings.setValue("device", self._device.currentText())
        self._settings.setValue("detector_type", self._detector_type.currentText())
        self._settings.setValue("conf_thresh", self._conf_thresh.value())
        self._settings.setValue("iou_thresh", self._iou_thresh.value())
        self._settings.setValue("image_size", self._image_size.value())
        self._settings.setValue("batch_size", self._batch_size.value())
        self._settings.setValue("workers", self._workers.value())
        self._settings.setValue("max_workers", self._max_workers.value())
        self._settings.setValue("tracker_max_age", self._tracker_max_age.value())
        self._settings.setValue("tracker_min_hits", self._tracker_min_hits.value())
        self._settings.setValue(
            "tracker_min_travel", self._tracker_min_travel.value()
        )
        self._settings.setValue(
            "tracker_iou_threshold", self._tracker_iou_threshold.value()
        )
        self._settings.setValue("tracker_reverse", self._tracker_reverse.isChecked())
        self._settings.setValue(
            "min_target_length", self._min_target_length.value()
        )
        self._settings.setValue(
            "max_target_length", self._max_target_length.value()
        )
        self._settings.setValue("distance_offset", self._distance_offset.value())
        self._settings.setValue("use_blur", self._use_blur.isChecked())
        self._settings.setValue(
            "use_multithreading", self._use_multithreading.isChecked()
        )
        self._settings.setValue(
            "export_summary_csv", self._export_summary_csv.isChecked()
        )
        self._settings.setValue(
            "export_detailed_csv", self._export_detailed_csv.isChecked()
        )
        self._settings.setValue("export_fc", self._export_fc.isChecked())
        self._settings.setValue("export_xml", self._export_xml.isChecked())
        self._settings.setValue("export_mot", self._export_mot.isChecked())

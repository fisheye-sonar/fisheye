"""Lightweight configuration types for the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from fisheye.enums import DetectorType


@dataclass
class PipelineConfig:
    input_path: Path
    checkpoint: Path
    start_frame: int = 0
    end_frame: int = -1
    output_dir: Path | None = None
    device: str = "cpu"
    detector_type: str = DetectorType.YOLOv5.value
    batch_size: int = 32
    workers: int = 0
    image_size: int = 896
    use_blur: bool = True
    dataset_use_multithreading: bool = False
    dataset_max_workers: int = 2
    use_multithreading: bool = False
    max_workers: int = 2
    conf: float = 0.10
    iou: float = 0.25
    upstream_direction: str = "left"
    distance_offset: float = 0.0
    min_target_length: float = 0.3
    max_target_length: float = 0.0
    tracker_max_age: int = 20
    tracker_min_hits: int = 2
    tracker_min_travel: int = 0
    tracker_iou_threshold: float = 0.001
    tracker_reverse: bool = False
    export_options: list[str] = field(
        default_factory=lambda: ["summary_csv", "detailed_csv", "fc", "xml"]
    )

    def validate(self) -> None:
        self.validate_shared()
        input_path = self.input_path.expanduser().resolve()
        if input_path.suffix.lower() not in (".aris", ".ddf"):
            raise ValueError(f"Expected .aris or .ddf file, got {input_path}")
        if not input_path.is_file():
            raise FileNotFoundError(f"ARIS file not found: {input_path}")

    def validate_shared(self) -> None:
        checkpoint = self.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        DetectorType(self.detector_type)

        if not self.export_options:
            raise ValueError("Enable at least one export format")

        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if self.start_frame < 0:
            raise ValueError("Start frame must be 0 or greater")
        if self.end_frame != -1 and self.end_frame < 0:
            raise ValueError("End frame must be -1 or greater")
        if self.end_frame != -1 and self.start_frame >= self.end_frame:
            raise ValueError("End frame must be greater than start frame")
        if self.workers < 0:
            raise ValueError("Workers must be 0 or greater")
        if self.dataset_max_workers < 1:
            raise ValueError("Dataset max workers must be at least 1")
        if self.max_workers < 1:
            raise ValueError("Max workers must be at least 1")
        if self.tracker_max_age < 0:
            raise ValueError("Tracker max age must be 0 or greater")
        if self.tracker_min_hits < 0:
            raise ValueError("Tracker min hits must be 0 or greater")
        if self.tracker_min_travel < 0:
            raise ValueError("Tracker min travel must be 0 or greater")
        if self.tracker_iou_threshold < 0.0:
            raise ValueError("Tracker IoU threshold must be 0 or greater")
        if self.image_size < 32:
            raise ValueError("Image size must be at least 32")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("IoU must be between 0 and 1")
        if self.min_target_length < 0:
            raise ValueError("Minimum target length must be 0 or greater")
        if self.max_target_length < 0:
            raise ValueError("Maximum target length must be 0 or greater")
        if self.max_target_length and self.max_target_length < self.min_target_length:
            raise ValueError(
                "Maximum target length must be 0 or greater than minimum target length"
            )

        if self.output_dir is not None:
            self.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)

    def with_input_path(self, input_path: Path) -> PipelineConfig:
        return replace(self, input_path=input_path)

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir.expanduser().resolve()
        return self.input_path.expanduser().resolve().parent

    @property
    def frame_range_suffix(self) -> str:
        return f"_{self.start_frame}_{self.end_frame}"


@dataclass
class PipelineResult:
    input_path: Path
    output_dir: Path
    total_seconds: float
    upstream_count: int = 0
    downstream_count: int = 0
    exported_paths: list[Path] = field(default_factory=list)

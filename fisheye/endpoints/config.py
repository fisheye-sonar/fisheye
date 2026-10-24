"""Runtime settings; model input semantics live in the exported bundle."""

from dataclasses import dataclass, field
from pathlib import Path
import math
from numbers import Integral


@dataclass
class TrackLengthConfig:
    """Cone-side margin in native pixels, then track observation/percentile rules."""

    enabled: bool = False
    edge_margin_px: float = 25.0
    min_track_points: int = 5
    percentile: float = 50.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("endpoints.track_length.enabled must be boolean")
        for name in ("edge_margin_px", "percentile"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"endpoints.track_length.{name} must be finite")
        if self.edge_margin_px < 0:
            raise ValueError(
                "endpoints.track_length.edge_margin_px must be nonnegative"
            )
        if not 0 <= self.percentile <= 100:
            raise ValueError(
                "endpoints.track_length.percentile must be between 0 and 100"
            )
        if (
            isinstance(self.min_track_points, bool)
            or not isinstance(self.min_track_points, Integral)
            or self.min_track_points < 1
        ):
            raise ValueError(
                "endpoints.track_length.min_track_points must be positive integer"
            )


def validate_capture_config(config):
    """Signed capture is an exact copy of the unblurred detector BGS channel."""
    if (
        config.use_blur
        or config.only_pos_bgs
        or not config.do_bg_subtract
        or config.return_unwarped
    ):
        raise ValueError(
            "bgs_signed_u8 requires detector use_blur=False, only_pos_bgs=False, "
            "do_bg_subtract=True and return_unwarped=False"
        )


@dataclass
class EndpointConfig:
    enabled: bool = False
    model: dict = field(default_factory=dict)
    device: str = "cpu"
    crop_batch_size: int = 1
    frame_store: dict = field(
        default_factory=lambda: {
            "ram_limit_mb": 512,
            "disk_limit_gb": 20,
            "directory": None,
        }
    )
    export_predictions: bool = True
    track_length: TrackLengthConfig = field(default_factory=TrackLengthConfig)

    def __post_init__(self):
        self.model = dict(self.model)
        self.frame_store = dict(self.frame_store)
        if not isinstance(self.track_length, TrackLengthConfig):
            self.track_length = TrackLengthConfig(**dict(self.track_length))
        if (
            isinstance(self.crop_batch_size, bool)
            or not isinstance(self.crop_batch_size, Integral)
            or self.crop_batch_size < 1
        ):
            raise ValueError("endpoints.crop_batch_size must be positive")
        if self.enabled and not self.model.get("bundle"):
            raise ValueError("endpoints.model.bundle is required when enabled")
        if set(self.model) - {"bundle"}:
            raise ValueError("Unknown endpoint model settings")
        if set(self.frame_store) - {"ram_limit_mb", "disk_limit_gb", "directory"}:
            raise ValueError("Unknown endpoint frame-store settings")
        for key, default in (("ram_limit_mb", 512), ("disk_limit_gb", 20)):
            value = self.frame_store.get(key, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"endpoints.frame_store.{key} must be nonnegative")

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        return cls(**dict(value or {}))

    def resolve_paths(self, root):
        for mapping, key in ((self.model, "bundle"), (self.frame_store, "directory")):
            if mapping.get(key):
                mapping[key] = str((Path(root) / mapping[key]).resolve())
        return self


class EndpointPipelineError(RuntimeError):
    """A failed endpoint clip must not be reported as an empty successful clip."""


class EndpointBatchError(RuntimeError):
    """All clips were attempted; retain successes while reporting failed clips."""

    def __init__(self, results, failures, failure_reports=(), report_errors=()):
        self.results = results
        self.failures = failures
        self.failure_reports = tuple(failure_reports)
        self.report_errors = tuple(report_errors)
        super().__init__(
            f"Endpoint processing failed for {len(failures)} of "
            f"{len(results) + len(failures)} clips; all clips were attempted. "
            f"Failure reports: {', '.join(map(str, failure_reports)) or 'unavailable'}"
        )

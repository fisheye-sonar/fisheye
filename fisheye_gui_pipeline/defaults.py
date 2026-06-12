"""Load GUI defaults from the same repo config files used by the normal pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from fisheye.configs import NMSConfig, YOLODatasetConfig
from fisheye.configs.inference import (
    ObjectDetectionConfig,
    TargetSizeConfig,
    TrackerConfig,
)
from fisheye.enums import DetectorType


@dataclass(frozen=True)
class RepoDefaults:
    checkpoint: Path
    detector_type: str
    device: str
    batch_size: int
    workers: int
    image_size: int
    use_blur: bool
    dataset_use_multithreading: bool
    dataset_max_workers: int
    inference_use_multithreading: bool
    inference_max_workers: int
    conf: float
    iou: float
    upstream_direction: str
    distance_offset: float
    min_target_length: float
    max_target_length: float
    tracker_max_age: int
    tracker_min_hits: int
    tracker_min_travel: int
    tracker_iou_threshold: float
    tracker_reverse: bool


def load_repo_defaults(project_root: Path | None = None) -> RepoDefaults:
    root = project_root or Path(__file__).resolve().parents[1]

    app_cfg = OmegaConf.load(root / "configs" / "config.yaml")
    defaults_list = OmegaConf.to_container(app_cfg.defaults, resolve=True)

    platform_name = "cpu"
    for item in defaults_list:
        if isinstance(item, dict) and "platform" in item:
            platform_name = str(item["platform"])
            break

    platform_cfg = OmegaConf.load(root / "configs" / "platform" / f"{platform_name}.yaml")

    dataset_defaults = YOLODatasetConfig()
    runtime_defaults = ObjectDetectionConfig()
    nms_defaults = NMSConfig()
    target_defaults = TargetSizeConfig()
    tracker_defaults = TrackerConfig()

    weights_path = (root / str(platform_cfg.model.weights)).resolve()

    return RepoDefaults(
        checkpoint=weights_path,
        detector_type=str(platform_cfg.model.type),
        device=str(platform_cfg.model.device),
        batch_size=int(platform_cfg.dataset.get("batch_size", dataset_defaults.batch_size)),
        workers=int(platform_cfg.dataset.get("workers", dataset_defaults.workers)),
        image_size=int(platform_cfg.dataset.get("img_size", dataset_defaults.img_size)),
        use_blur=bool(platform_cfg.dataset.get("use_blur", dataset_defaults.use_blur)),
        dataset_use_multithreading=bool(
            platform_cfg.dataset.get(
                "use_multithreading", dataset_defaults.use_multithreading
            )
        ),
        dataset_max_workers=int(
            platform_cfg.dataset.get("max_workers", dataset_defaults.max_workers)
        ),
        inference_use_multithreading=bool(
            platform_cfg.inference.get(
                "use_multithreading", runtime_defaults.use_multithreading
            )
        ),
        inference_max_workers=int(
            platform_cfg.inference.get("max_workers", runtime_defaults.max_workers)
        ),
        conf=float(platform_cfg.inference.get("conf", nms_defaults.conf)),
        iou=float(platform_cfg.inference.get("iou", nms_defaults.iou)),
        upstream_direction=str(app_cfg.get("upstream_direction", "left")),
        distance_offset=float(app_cfg.get("distance_offset", 0.0)),
        min_target_length=float(
            app_cfg.get("target_size", {}).get(
                "min_length", target_defaults.min_length
            )
        ),
        max_target_length=float(
            app_cfg.get("target_size", {}).get(
                "max_length", target_defaults.max_length
            )
        ),
        tracker_max_age=int(tracker_defaults.max_age),
        tracker_min_hits=int(tracker_defaults.min_hits),
        tracker_min_travel=int(tracker_defaults.min_travel),
        tracker_iou_threshold=float(tracker_defaults.iou_threshold),
        tracker_reverse=bool(tracker_defaults.reverse),
    )

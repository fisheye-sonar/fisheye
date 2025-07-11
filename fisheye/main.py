import argparse
import time
from pathlib import Path

import hydra
import structlog
from omegaconf import DictConfig

from fisheye.common.logging import setup_logging
from fisheye.common.system import check_disk_space, generate_job_id
from fisheye.configs import YOLOv5ModelConfig, ObjectDetectionConfig, YOLODatasetConfig
from fisheye.enums import ExportType
from fisheye.export import save_to_disk, parse_export_options
from fisheye.pipelines.pipeline import DetectTrackCountPipeline
from fisheye.version import __app_version__, get_version_from_detector

job_id = generate_job_id()
setup_logging(file_logging=True, job_id=job_id)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    input_path = cfg.input_path
    output_dir = cfg.output_dir
    export_options = cfg.export_options

    export_types = parse_export_options(export_options)

    # Use specific platform config
    platform_cfg = cfg.platform
    dataset_config = platform_cfg.dataset
    model_config = platform_cfg.model

    check_disk_space(path=output_dir)  # Make sure there's enough space to store results
    project_root = Path(__file__).resolve().parents[1]

    weights, device = model_config.weights, model_config.device
    model_path = str((project_root / weights).resolve())

    # Bind job ID, app, and detector version to logger
    logger = structlog.get_logger().bind(
        job_id=job_id,
        app_version=__app_version__,
        detector_version=get_version_from_detector(model_path),
    )

    # Build configs
    yolo_cfg = YOLOv5ModelConfig(weights=model_path, device=device)
    inference_config = dict(platform_cfg.inference)
    inference_config["model"] = yolo_cfg
    detection_cfg = ObjectDetectionConfig(**inference_config)
    dataset_cfg = YOLODatasetConfig(**dataset_config)

    start_time = time.time()
    logger.info("inference_started", start_time=start_time)

    results = DetectTrackCountPipeline(
        detector_cfg=detection_cfg, dataset_cfg=dataset_cfg
    ).run(input_path, output_dir, export_types, job_id)

    if ExportType.SUMMARY_CSV in export_types:
        save_to_disk(
            results, output_dir, export_types=ExportType.SUMMARY_CSV, job_id=job_id
        )

    end_time = time.time()
    logger.info(
        "inference_completed",
        inference_duration_sec=end_time - start_time,
        num_files_processed=len(results),
    )

    return results


if __name__ == "__main__":
    main()

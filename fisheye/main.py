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
from fisheye.export import save_to_disk
from fisheye.pipelines.pipeline import DetectTrackCountPipeline
from fisheye.version import __app_version__, get_version_from_detector

job_id = generate_job_id()
setup_logging(file_logging=True, job_id=job_id)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    input_path = cfg.input_path
    output_dir = cfg.output_dir
    export_options = cfg.export_options

    parts = [v.strip().upper() for v in export_options]
    export_types = []
    for p in parts:
        try:
            export_types.append(ExportType[p])
        except KeyError as e:
            raise argparse.ArgumentTypeError(f"Invalid export type: {e.args[0]}")

    # Get platform specific config
    base_config = cfg.platform
    dataset_config = base_config.dataset
    model_config = base_config.model

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

    # Get config to run object detector
    yolo_cfg = YOLOv5ModelConfig(weights=model_path, device=device)
    task_config = dict(base_config.inference)
    task_config["model"] = yolo_cfg

    detection_cfg = ObjectDetectionConfig(**task_config)

    start_time = time.time()
    logger.info("inference_started", start_time=start_time)

    # Dataset prep
    dataset_cfg = YOLODatasetConfig(**dataset_config)
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

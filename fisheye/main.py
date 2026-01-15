import platform
import time
from pathlib import Path

import hydra
import structlog
import torch.multiprocessing as mp
from omegaconf import DictConfig

from fisheye.common.file_system import is_valid_dir
from fisheye.common.logging import setup_logging
from fisheye.common.system import check_disk_space, generate_job_id
from fisheye.configs import (
    ObjectDetectionConfig,
    YOLODatasetConfig,
    get_detector_config,
    get_length_model_config,
)
from fisheye.detect.factory import DETECTOR_CLASS_REGISTRY
from fisheye.enums import ExportType, DetectorType, LengthEstimatorType
from fisheye.export import save_to_disk, parse_export_options
from fisheye.pipelines import ObjectDetectionPipeline
from fisheye.pipelines.pipeline import DetectTrackCountPipeline
from fisheye.version import __app_version__, get_version_from_detector


def run_pipeline(cfg: DictConfig):
    job_id = generate_job_id()
    setup_logging(file_logging=True, job_id=job_id)

    input_path = cfg.input_path
    output_dir = cfg.output_dir
    export_options = cfg.export_options
    upstream_direction = cfg.upstream_direction
    distance_offset = cfg.distance_offset

    export_types = parse_export_options(export_options)

    # Use specific platform config
    platform_cfg = cfg.platform
    dataset_config = platform_cfg.dataset
    model_config = platform_cfg.model

    check_disk_space(
        path=output_dir if output_dir else input_path
    )  # Make sure there's enough space to store results
    project_root = Path(__file__).resolve().parents[1]

    weights, device = model_config.weights, model_config.device
    resolved_weights_path = str((project_root / weights).resolve())

    # Bind job ID, app, and detector version to logger
    logger = structlog.get_logger().bind(
        job_id=job_id,
        app_version=__app_version__,
        detector_version=get_version_from_detector(resolved_weights_path),
    )

    # Build config + model for detection
    detector_type = DetectorType(platform_cfg.model.type)
    detector_cfg = get_detector_config(detector_type, **platform_cfg.model)
    detector_cfg.weights = resolved_weights_path
    detector = DETECTOR_CLASS_REGISTRY[detector_type](detector_cfg)

    # Build runtime configs
    runtime_config = dict(platform_cfg.inference)
    if "length_config" in runtime_config:
        length_cfg_dict = dict(runtime_config["length_config"])
        if length_cfg_dict["weights"]:
            length_cfg_dict["weights"] = (
                project_root / length_cfg_dict["weights"]
            ).resolve()
        model_type = length_cfg_dict.pop("type", LengthEstimatorType.UNET.value)
        runtime_config["length_config"] = get_length_model_config(
            model_type, **length_cfg_dict
        )

    runtime_config["model"] = detector_cfg

    # Build dataset configs
    dataset_cfg = YOLODatasetConfig(**dataset_config)

    start_time = time.time()
    logger.info("inference_started", start_time=start_time)

    # Build out individual pipeline(s)
    detector_pipe = ObjectDetectionPipeline(
        model=detector, config=ObjectDetectionConfig(**runtime_config)
    )

    results = DetectTrackCountPipeline(detector_pipe, dataset_cfg=dataset_cfg).run(
        input_path,
        output_dir,
        export_types,
        job_id,
        upstream_direction,
        distance_offset,
    )

    if results:
        if ExportType.SUMMARY_CSV in export_types:
            if not output_dir:
                output_dir = (
                    Path(input_path)
                    if is_valid_dir(input_path)
                    else Path(input_path).parent
                )

            save_to_disk(
                results,
                output_dir,
                export_types=ExportType.SUMMARY_CSV,
                job_id=job_id,
                distance_offset=distance_offset,
                upstream_direction=upstream_direction,
            )

    end_time = time.time()
    logger.info(
        "inference_completed",
        inference_duration_sec=end_time - start_time,
        num_files_processed=len(results),
    )

    return results


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    return run_pipeline(cfg)


if __name__ == "__main__":
    # TODO (MVH): This check is a bit rough since we don't have the config yet, but 'spawn' is generally safer for
    #  Windows + multiprocessing anyways
    if platform.system() == "Windows":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    main()

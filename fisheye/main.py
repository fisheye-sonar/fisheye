import platform
from pathlib import Path

import hydra
import structlog
import torch.multiprocessing as mp
from omegaconf import DictConfig

from fisheye.builder import PipelineFactory
from fisheye.common.system import check_disk_space, generate_job_id
from fisheye.common.logging import setup_logging
from fisheye.export import parse_export_options
from fisheye.runner import PipelineRunner
from fisheye.version import __app_version__, get_version_from_detector


def run_job(cfg: DictConfig):
    job_id = generate_job_id()
    setup_logging(file_logging=True, job_id=job_id)

    input_path = cfg.input_path
    output_dir = cfg.output_dir
    export_options = cfg.export_options

    # Parse export options
    export_types = parse_export_options(export_options)

    # Use specific platform config
    platform_cfg = cfg.platform

    # Check disk space
    check_disk_space(path=output_dir if output_dir else input_path)

    project_root = Path(__file__).resolve().parents[1]

    # Build components
    detector, resolved_weights_path, detector_cfg = PipelineFactory.build_detector(
        platform_cfg, project_root
    )

    # Bind job ID, app, and detector version to logger
    structlog.get_logger().bind(
        job_id=job_id,
        app_version=__app_version__,
        detector_version=get_version_from_detector(resolved_weights_path),
    )

    dataset_cfg = PipelineFactory.build_dataset_config(platform_cfg.dataset)

    runtime_config = PipelineFactory.build_runtime_config(
        platform_cfg, project_root, detector_cfg
    )

    pipeline = PipelineFactory.build_pipeline(detector, runtime_config, dataset_cfg)

    # Run
    runner = PipelineRunner(pipeline)
    return runner.run(
        input_path,
        output_dir,
        export_types,
        job_id,
        cfg.upstream_direction,
        cfg.distance_offset,
    )


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    return run_job(cfg)


if __name__ == "__main__":
    # TODO (MVH): This check is a bit rough since we don't have the config yet, but 'spawn' is generally safer for
    #  Windows + multiprocessing anyways
    if platform.system() == "Windows":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    main()

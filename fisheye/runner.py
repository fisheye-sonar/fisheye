import time
from pathlib import Path
from typing import List, Union

import structlog
from fisheye.builder import PipelineFactory
from fisheye.common.file_system import is_valid_dir
from fisheye.common.logging import setup_logging
from fisheye.common.system import check_disk_space, generate_job_id
from fisheye.enums import ExportType
from fisheye.export import parse_export_options, save_to_disk
from fisheye.pipelines.pipeline import DetectTrackCountPipeline
from fisheye.version import __app_version__, get_version_from_detector
from omegaconf import DictConfig

logger = structlog.get_logger()


class PipelineRunner:
    """Orchestrator for the pipeline execution."""

    def __init__(self, pipeline: DetectTrackCountPipeline):
        self.pipeline = pipeline

    def run(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        export_types: List[ExportType],
        job_id: str,
        upstream_direction: str,
        distance_offset: float,
    ) -> List:
        """Run the pipeline and handle result orchestration."""

        start_time = time.time()
        logger.info("inference_started", start_time=start_time)

        results = self.pipeline.run(
            input_path,
            output_dir,
            export_types,
            job_id,
            upstream_direction,
            distance_offset,
        )

        if results and ExportType.SUMMARY_CSV in export_types:
            self._save_summary_csv(
                results,
                input_path,
                output_dir,
                job_id,
                upstream_direction,
                distance_offset,
            )

        end_time = time.time()
        logger.info(
            "inference_completed",
            inference_duration_sec=end_time - start_time,
            num_files_processed=len(results),
        )

        return results

    def _save_summary_csv(
        self,
        results,
        input_path,
        output_dir,
        job_id,
        upstream_direction,
        distance_offset,
    ):
        """Save the summary CSV if enabled."""
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


def run_job(cfg: DictConfig):
    """Run the job defined by the configuration."""
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

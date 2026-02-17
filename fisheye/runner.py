import time
from pathlib import Path
from typing import List, Union

import structlog

from fisheye.common.file_system import is_valid_dir
from fisheye.enums import ExportType
from fisheye.export import save_to_disk
from fisheye.pipelines.pipeline import DetectTrackCountPipeline

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

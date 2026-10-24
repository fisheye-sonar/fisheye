from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional, List, Union
from uuid import uuid4

import structlog

from fisheye.boxes import (
    mean_bbox_width_yolo_to_image,
    median_bbox_width_yolo_to_image,
    std_bbox_width_yolo_to_image,
)
from fisheye.common.generic import safe_execution
from fisheye.endpoints.config import (
    EndpointBatchError,
    EndpointConfig,
    EndpointPipelineError,
    validate_capture_config,
)
from fisheye.common.file_system import get_valid_files
from fisheye.configs import YOLODatasetConfig
from fisheye.configs.inference import (
    TrackerConfig,
    LengthEstimationConfig,
    TargetSizeConfig,
)
from fisheye.count.counter import Count
from fisheye.enums import ExportType, UpstreamDirectionTypes
from fisheye.export import save_to_disk, MOTExporter
from fisheye.format import (
    tracker_output_to_dict_rows,
    dict_rows_to_mot_format,
    format_single_crossing,
)
from fisheye.lengths.processor import LengthProcessor
from fisheye.pipelines import ObjectDetectionPipeline
from fisheye.track.tracker import run_tracker

logger = structlog.getLogger(__name__)


class DetectTrackCountPipeline:
    """Pipeline for detection, tracking, and counting."""

    def __init__(
        self,
        detect_pipe: Optional[ObjectDetectionPipeline] = None,
        tracker_cfg: Optional[TrackerConfig] = None,
        dataset_cfg: YOLODatasetConfig = None,
        target_size: TargetSizeConfig = None,
    ):
        self.detect_pipe = detect_pipe
        self.tracker_cfg = tracker_cfg if tracker_cfg else TrackerConfig()
        self.dataset_cfg = dataset_cfg if dataset_cfg else YOLODatasetConfig()
        self.length_cfg = LengthEstimationConfig()
        self.target_size = target_size or TargetSizeConfig()
        self.endpoint_config = EndpointConfig.from_value(
            getattr(detect_pipe, "endpoint_config", None)
        )
        self.endpoint_processor = None
        if self.endpoint_config.enabled:
            if self.dataset_cfg.return_unwarped:
                raise ValueError("Endpoint inference requires warped image coordinates")
            from fisheye.endpoints.processor import EndpointProcessor

            self.endpoint_processor = EndpointProcessor(self.endpoint_config)
            self.dataset_cfg = replace(
                self.dataset_cfg,
                capture_endpoint_frames=True,
            )
            validate_capture_config(self.dataset_cfg)

    @safe_execution(
        default_return=[], max_retries=3, delay=2, passthrough=(EndpointPipelineError,)
    )
    def _run(self, file, output_dir, *args, **kwargs):
        if self.endpoint_processor is None:
            return self._run_clip(file, output_dir, *args, **kwargs)
        try:
            with self.endpoint_processor.session(str(Path(file).resolve())) as session:
                self.detect_pipe.endpoint_session = session
                try:
                    return self._run_clip(file, output_dir, *args, **kwargs)
                finally:
                    self.detect_pipe.endpoint_session = None
        except Exception as exc:
            logger.error("endpoint_clip_failed", file=str(file), error=str(exc))
            raise EndpointPipelineError(
                f"Endpoint processing failed for {file}: {exc}"
            ) from exc

    def _run_clip(
        self,
        file: Union[Path, List[Path]],
        output_dir: Union[str, Path],
        export_types: Optional[List[ExportType]] = None,
        job_id: Optional[str] = None,
        upstream_direction: UpstreamDirectionTypes = UpstreamDirectionTypes.LEFT,
        distance_offset: Union[int, float] = 0.0,
        file_index: Optional[int] = None,
    ) -> List:

        if not output_dir:
            output_dir = file.parent

        # Shallow copy of YOLODatasetConfig with updated fields
        self.dataset_cfg = replace(
            self.dataset_cfg, filepath=file, start_frame=0, end_frame=0
        )

        self.detect_pipe.load_dataset(self.dataset_cfg)
        low_preds, high_preds, low_length_estimates, high_length_estimates = (
            self.detect_pipe()
        )
        metadata = self.detect_pipe.metadata

        tracker_options = {}
        if self.endpoint_processor is not None:
            tracker_options = {
                "frame_ids": self.detect_pipe.endpoint_session.frame_ids,
                "record_detection_matches": True,
            }
        tracker_output = run_tracker(
            low_preds,  # normalized xyxy in original warped image coordinates
            high_preds,
            metadata.image_meter_width,
            metadata.image_meter_height,
            self.tracker_cfg,
            min_length=self.target_size.min_length,
            **tracker_options,
        )

        if self.endpoint_processor is not None:
            self.endpoint_processor.process(
                self.detect_pipe.endpoint_session,
                tracker_output,
                low_preds,
                high_preds,
                metadata,
                Path(output_dir),
                Path(file).stem,
                detector_provenance=self.detect_pipe.endpoint_detection_provenance(),
                tracker_config={
                    **asdict(self.tracker_cfg),
                    "target_size": asdict(self.target_size),
                },
            )
            len_outputs = self.endpoint_processor.last_length_outputs
        else:
            len_outputs = self._estimate_lengths(
                tracker_output,
                low_length_estimates,
                high_length_estimates,
            )

        if len_outputs:
            num_fish_with_lengths = sum(
                1 for v in len_outputs.values() if v is not None
            )
            logger.info(
                "length_estimation_complete",
                total_fish=len(len_outputs),
                fish_with_valid_lengths=num_fish_with_lengths,
            )

        formatted_yolo_tracks = tracker_output_to_dict_rows(asdict(tracker_output))

        if export_types is None:
            export_types_list = []

        elif isinstance(export_types, ExportType):
            export_types_list = [export_types]

        else:
            export_types_list = export_types

        if ExportType.MOT in export_types_list:
            mot_tracks = dict_rows_to_mot_format(
                formatted_yolo_tracks, metadata.xdim, metadata.ydim
            )
            MOTExporter(output_dir=output_dir, filename=file.stem).export(mot_tracks)

        (left_count, right_count), crossing_frames = Count().count(
            formatted_yolo_tracks
        )

        if crossing_frames and (left_count or right_count):
            formatted_crossings = [
                format_single_crossing(
                    filename=Path(file).name,
                    metadata=metadata,
                    source_path=str(Path(file).resolve()),
                    track_id=track_id,
                    frame=frame,
                    bbox=bbox,
                    upstream_direction=upstream_direction,
                    crossing_direction="left",
                    len_outputs=len_outputs,
                    file_index=file_index,
                )
                for track_id, frame, bbox in crossing_frames["left"]
            ] + [
                format_single_crossing(
                    filename=Path(file).name,
                    metadata=metadata,
                    source_path=str(Path(file).resolve()),
                    track_id=track_id,
                    frame=frame,
                    bbox=bbox,
                    upstream_direction=upstream_direction,
                    crossing_direction="right",
                    len_outputs=len_outputs,
                    file_index=file_index,
                )
                for track_id, frame, bbox in crossing_frames["right"]
            ]

            # TODO (MHV): Try/except is temporary until this logic has been tested more vigorously
            try:
                # Extract unique track IDs
                unique_ids = {row["id"] for row in formatted_yolo_tracks}
                num_tracks = len(unique_ids)

                # Extract all bboxes from the formatted crossings
                all_bboxes = [f["bbox"] for f in formatted_crossings]
                std_bbox_width = std_bbox_width_yolo_to_image(
                    all_bboxes, metadata.image_meter_width
                )
                avg_bbox_width = mean_bbox_width_yolo_to_image(
                    all_bboxes, metadata.image_meter_width
                )
                median_bbox_width = median_bbox_width_yolo_to_image(
                    all_bboxes, metadata.image_meter_width
                )

                # Log stats for current ARIS file
                logger.info(
                    "processed_file_stats",
                    file_index=file_index,
                    num_counts=len(formatted_crossings),
                    num_tracks=num_tracks,
                    avg_bbox_width_meters=avg_bbox_width,
                    median_bbox_width_meters=median_bbox_width,
                    std_bbox_width_meters=std_bbox_width,
                )

            except Exception:
                # Silently skip stats if calculation fails.
                pass

        else:
            formatted_crossings = [
                format_single_crossing(
                    filename=Path(file).name,
                    metadata=metadata,
                    source_path=str(Path(file).resolve()),
                    file_index=file_index,
                )
            ]

            logger.warning("no_counts", file_path=str(file), file_index=file_index)

        remaining_export_types = [
            et
            for et in export_types_list
            if et != ExportType.MOT and (et != ExportType.SUMMARY_CSV)
        ]
        save_to_disk(
            [formatted_crossings],
            output_dir,
            export_types=remaining_export_types,
            job_id=job_id,
            distance_offset=distance_offset,
            upstream_direction=upstream_direction,
        )

        return formatted_crossings

    def _estimate_lengths(
        self,
        tracker_output,
        low_length_estimates,
        high_length_estimates,
    ):
        """Estimate fish lengths.

        Returns:
            Dict mapping fish_id to length estimate dict (or None if estimation disabled)
        """

        if not hasattr(self.detect_pipe, "apply_length_estimates_batchwise"):
            logger.warning(
                "skip_length_estimation",
                message="Length estimator not configured in detection pipeline",
            )
            return {}

        if not self.detect_pipe.apply_length_estimates_batchwise:
            logger.warning(
                "skip_length_estimation",
                message="Length estimation disabled (batchwise mode required)",
            )
            return {}

        if not low_length_estimates and not high_length_estimates:
            logger.warning(
                "skip_length_estimation",
                message="No length estimates available from detection pipeline",
            )
            return {}

        processor = LengthProcessor(self.length_cfg, self.detect_pipe.metadata)

        frames_preds = asdict(tracker_output)["frames"]
        len_outputs = processor.process_from_tracks(
            frames_preds,
            low_length_estimates,
            high_length_estimates,
            self.dataset_cfg.batch_size,
        )

        len_outputs_dict = {
            fish_id: asdict(estimate) if estimate is not None else None
            for fish_id, estimate in len_outputs.items()
        }

        return len_outputs_dict

    def run(
        self,
        file: Union[str, Path, List[Union[str, Path]]],
        output_dir: Union[str, Path],
        export_types: Optional[List[ExportType]] = None,
        job_id: Optional[str] = None,
        upstream_direction: UpstreamDirectionTypes = UpstreamDirectionTypes.LEFT,
        distance_offset: Union[int, float] = 0.0,
    ) -> Union[List[List[dict]], List[dict]]:
        """Run preprocessing, detection, tracking, and counting on frames.

        Args:
            file (List[str] | str): File(s) to process. Must be a path to an ARIS file or a directory holding ARIS files
            output_dir (str): Output directory to save results to
            export_types (Optional[List[ExportType]]): List of ExportType objects to export to
            job_id (Optional[str]): Job ID
            upstream_direction (Optional[UpstreamDirectionTypes]): Upstream direction

        Returns:
            dict: Tracking results and counts.
        """

        valid_files = get_valid_files(
            file,
            output_dir,
            skip_existing_fc=getattr(self, "endpoint_processor", None) is None,
        )
        if not valid_files:
            logger.error(
                f"Unable to process valid files. Please verify that the file path is correct and that the "
                f"file hasn't already been processed."
            )
            return []

        logger.info("files_discovered", num_files=len(valid_files))

        results, failures = [], []
        for file_index, source in enumerate(valid_files):
            try:
                result = self._run(
                    source,
                    output_dir,
                    export_types,
                    job_id,
                    upstream_direction,
                    distance_offset,
                    file_index,
                )
            except EndpointPipelineError as exc:
                failures.append(
                    {
                        "source_path": str(Path(source).resolve()),
                        "file_index": file_index,
                        "error_type": type(exc.__cause__ or exc).__name__,
                        "message": str(exc),
                    }
                )
                logger.error(
                    "endpoint_file_failed_continuing",
                    file=str(source),
                    file_index=file_index,
                    error=str(exc),
                )
            else:
                results.append(result)

        if failures:
            from fisheye.endpoints.processor import atomic_json

            # Match normal per-clip output placement when no shared directory
            # was supplied. Use a unique basename to retain previous failures.
            directories = {
                Path(output_dir) if output_dir else Path(f["source_path"]).parent
                for f in failures
            }
            basename = f"endpoint_failures_{uuid4().hex}.json"
            reports, report_errors = [], []
            for directory in sorted(directories):
                path = directory / basename
                try:
                    atomic_json(
                        path,
                        {
                            "schema_version": 1,
                            "status": "partial_failure",
                            "job_id": job_id,
                            "attempted_clips": len(valid_files),
                            "successful_clips": len(results),
                            "failed_clips": len(failures),
                            "failures": failures,
                        },
                    )
                except OSError as exc:
                    # A full output disk must not discard successful results
                    # before the runner can attempt their combined summary.
                    report_errors.append({"path": str(path), "message": str(exc)})
                    logger.error(
                        "endpoint_failure_report_failed", path=str(path), error=str(exc)
                    )
                else:
                    reports.append(str(path))
            raise EndpointBatchError(results, failures, reports, report_errors)

        return results

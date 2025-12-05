from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional, List, Union

import structlog

from fisheye.boxes import (
    mean_bbox_width_yolo_to_image,
    median_bbox_width_yolo_to_image,
    std_bbox_width_yolo_to_image,
)
from fisheye.common.generic import safe_execution
from fisheye.common.file_system import get_valid_files
from fisheye.configs import YOLODatasetConfig
from fisheye.configs.inference import TrackerConfig, LengthConfig
from fisheye.count.counter import Count
from fisheye.enums import ExportType, UpstreamDirectionTypes
from fisheye.export import save_to_disk, MOTExporter
from fisheye.export.constants import FC_DEFAULT_LENGTH_CM
from fisheye.format import tracker_output_to_dict_rows, dict_rows_to_mot_format
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
    ):
        self.detect_pipe = detect_pipe
        self.tracker_cfg = tracker_cfg if tracker_cfg else TrackerConfig()
        self.dataset_cfg = dataset_cfg if dataset_cfg else YOLODatasetConfig()
        self.length_cfg = LengthConfig()

    @safe_execution(default_return=[], max_retries=3, delay=2)
    def _run(
        self,
        file: Union[Path, List[Path]],
        output_dir: Union[str, Path],
        export_types: Optional[List[ExportType]] = None,
        job_id: Optional[str] = None,
        upstream_direction: UpstreamDirectionTypes = UpstreamDirectionTypes.LEFT,
        distance_offset: Union[int, float] = 0.0,
    ) -> List:

        if not output_dir:
            output_dir = file.parent

        logger.info("file_processing_started", file_path=str(file))
        # Shallow copy of YOLODatasetConfig with updated fields
        self.dataset_cfg = replace(
            self.dataset_cfg, filepath=file, start_frame=0, end_frame=0
        )

        self.detect_pipe.load_dataset(self.dataset_cfg)
        low_preds, high_preds, low_length_estimates, high_length_estimates = (
            self.detect_pipe()
        )
        metadata = self.detect_pipe.metadata

        tracker_output = run_tracker(
            low_preds,  # xyxy format relative to the original image pixel space
            high_preds,  # xyxy format relative to the original image pixel space
            metadata.image_meter_width,
            metadata.image_meter_height,
            self.tracker_cfg,
        )

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
            # Using the same naming conventions in ARISFish Software
            formatted_crossings = [
                {
                    "Source.Name": Path(file).name,
                    "Frame#": frame,
                    "Dir": "Up" if upstream_direction == "left" else "Down",
                    "ID": track_id,
                    "bbox": bbox,  # [x_center, y_center, width, height] relative to original image space
                    "metadata": metadata,
                    "L(cm)": round(
                        len_outputs[track_id].get(
                            "filtered_lengths_cm", FC_DEFAULT_LENGTH_CM
                        ),
                        2,
                    ),
                }
                for track_id, frame, bbox in crossing_frames["left"]
            ] + [
                {
                    "Source.Name": Path(file).name,
                    "Frame#": frame,
                    "Dir": "Up" if upstream_direction == "right" else "Down",
                    "ID": track_id,
                    "bbox": bbox,  # [x_center, y_center, width, height] relative to original image space
                    "metadata": metadata,
                    "L(cm)": round(
                        len_outputs[track_id].get(
                            "filtered_lengths_cm", FC_DEFAULT_LENGTH_CM
                        ),
                        2,
                    ),
                }
                for track_id, frame, bbox in crossing_frames["right"]
            ]

            # TODO (MHV): Try/except is temporary until this logic has been tested more vigorously
            try:
                formated_yolo_tracks = None
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
                {
                    "Source.Name": Path(file).name,
                    "Frame#": None,
                    "Dir": None,
                    "ID": None,
                    "bbox": None,
                    "metadata": metadata,
                    "L(cm)": None,
                }
            ]

            logger.warning("no_counts", file_path=str(file))

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
        # Check if length estimation is enabled
        if not hasattr(self.detect_pipe, "apply_length_estimates_batchwise"):
            logger.debug("Length estimation not configured in detection pipeline")
            return {}

        if not self.detect_pipe.apply_length_estimates_batchwise:
            logger.debug("Length estimation disabled (batchwise mode required)")
            return {}

        # Check if we have length estimates from detection
        if not low_length_estimates and not high_length_estimates:
            logger.debug("No length estimates available from detection pipeline")
            return {}

        processor = LengthProcessor(self.length_cfg, self.detect_pipe.metadata)

        frames_preds = asdict(tracker_output)["frames"]
        len_outputs = processor.process_from_tracks(
            frames_preds,
            low_length_estimates,
            high_length_estimates,
            self.dataset_cfg.batch_size,
        )

        # Convert LengthEstimate dataclasses to dicts for compatibility
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

        valid_files = get_valid_files(file, output_dir)
        if not valid_files:
            logger.error(
                f"Unable to process valid files. Please verify that the file path is correct and that the "
                f"file hasn't already been processed."
            )
            return []

        results = [
            self._run(
                f, output_dir, export_types, job_id, upstream_direction, distance_offset
            )
            for f in valid_files
        ]

        return results

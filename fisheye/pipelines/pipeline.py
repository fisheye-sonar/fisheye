from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional, List, Union

import structlog

from fisheye.boxes import run_nms, normalize_boxes_for_tracking
from fisheye.common.generic import safe_execution
from fisheye.common.file_system import get_valid_files
from fisheye.configs import ObjectDetectionConfig, YOLODatasetConfig
from fisheye.configs.inference import TrackerConfig, NMSConfig
from fisheye.count.counter import Count
from fisheye.enums import ExportType, UpstreamDirectionTypes
from fisheye.export import save_to_disk, to_mot_txt
from fisheye.format import tracker_output_to_dict_rows, dict_rows_to_mot_format
from fisheye.pipelines import ObjectDetectionPipeline
from fisheye.track.tracker import run_tracker

logger = structlog.getLogger(__name__)


class DetectTrackCountPipeline:
    """Pipeline for detection, tracking, and counting."""

    def __init__(
        self,
        detector_cfg: Optional[ObjectDetectionConfig] = None,
        tracker_cfg: Optional[TrackerConfig] = None,
        dataset_cfg: YOLODatasetConfig = None,
    ):
        self.detector_cfg = detector_cfg if detector_cfg else ObjectDetectionConfig()
        self.tracker_cfg = tracker_cfg if tracker_cfg else TrackerConfig()
        self.nms_config = NMSConfig()
        self.dataset_cfg = dataset_cfg if dataset_cfg else YOLODatasetConfig()

    @safe_execution(default_return=[], max_retries=3, delay=2)
    def _run(
        self,
        file: Union[Path, List[Path]],
        output_dir: Union[str, Path],
        export_types: Optional[List[ExportType]] = None,
        job_id: Optional[str] = None,
        upstream_direction: UpstreamDirectionTypes = UpstreamDirectionTypes.LEFT,
    ) -> List:

        if output_dir is None:
            output_dir = file.parent

        logger.info("file_processing_started", file_path=str(file))
        # Shallow copy of YOLODatasetConfig with updated fields
        self.dataset_cfg = replace(
            self.dataset_cfg, filepath=file, start_frame=0, end_frame=0
        )

        detector = ObjectDetectionPipeline(self.detector_cfg, self.dataset_cfg)
        detections = detector()

        metadata = detector.metadata

        # Get low confidence for ByteTrack
        self.nms_config.conf = 0.1
        low_output = run_nms(
            detections.pred_bboxes,  # xyxy format relative to YOLO pixel space
            metadata.image_meter_width,
            detections.width,
            self.dataset_cfg.batch_size,
            self.nms_config,
        )

        # Get high confidence for ByteTrack
        self.nms_config.conf = 0.3
        high_output = run_nms(
            detections.pred_bboxes,  # xyxy format relative to YOLO pixel space
            metadata.image_meter_width,
            detections.width,
            self.dataset_cfg.batch_size,
            self.nms_config,
        )

        # Prepare bounding boxes for tracking pipeline
        low_preds, og_width, og_height = normalize_boxes_for_tracking(
            detections.image_shape,
            low_output,  # xyxy format relative to YOLO pixel space
            detections.width,
            detections.height,
            batch_size=self.dataset_cfg.batch_size,
        )
        high_preds, og_width, og_height = normalize_boxes_for_tracking(
            detections.image_shape,
            high_output,  # xyxy format relative to YOLO pixel space
            detections.width,
            detections.height,
            batch_size=self.dataset_cfg.batch_size,
        )

        tracker_output = run_tracker(
            low_preds,  # xyxy format relative to the original image pixel space
            high_preds,  # xyxy format relative to the original image pixel space
            metadata.image_meter_width,
            metadata.image_meter_height,
            self.tracker_cfg,
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
            to_mot_txt(mot_tracks, output_dir, file.stem)

        (left_count, right_count), crossing_frames = Count().count(
            formatted_yolo_tracks
        )

        if crossing_frames and (left_count or right_count):
            formatted_crossings = [
                {
                    "fish_id": track,
                    "direction": "Up" if upstream_direction == "left" else "Down",
                    "frame_id": frame,
                    "file_name": Path(file).name,
                    "bbox": bbox,  # [x_center, y_center, width, height] relative to original image space
                    "metadata": metadata,
                }
                for track, frame, bbox in crossing_frames["left"]
            ] + [
                {
                    "fish_id": track,
                    "direction": "Up" if upstream_direction == "right" else "Down",
                    "frame_id": frame,
                    "file_name": Path(file).name,
                    "bbox": bbox,  # [x_center, y_center, width, height] relative to original image space
                    "metadata": metadata,
                }
                for track, frame, bbox in crossing_frames["right"]
            ]
        else:
            formatted_crossings = [
                {
                    "fish_id": None,
                    "direction": None,
                    "frame_id": None,
                    "file_name": Path(file).name,
                    "bbox": None,
                    "metadata": metadata,
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
        )

        return formatted_crossings

    def run(
        self,
        file: Union[str, Path, List[Union[str, Path]]],
        output_dir: Union[str, Path],
        export_types: Optional[List[ExportType]] = None,
        job_id: Optional[str] = None,
        upstream_direction: UpstreamDirectionTypes = UpstreamDirectionTypes.LEFT,
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
            self._run(f, output_dir, export_types, job_id, upstream_direction)
            for f in valid_files
        ]

        return results

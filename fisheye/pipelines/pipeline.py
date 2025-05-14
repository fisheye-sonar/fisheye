import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List

from fisheye.boxes import run_nms, normalize_boxes_for_tracking
from fisheye.common.generic import safe_execution
from fisheye.configs import ObjectDetectionConfig, YOLODatasetConfig
from fisheye.configs.inference import TrackerConfig, NMSConfig
from fisheye.count.counter import Count
from fisheye.enums import ExportType
from fisheye.export import save_to_disk
from fisheye.format import tracker_output_to_mot
from fisheye.pipelines import ObjectDetectionPipeline
from fisheye.track.tracker import run_tracker

logger = logging.getLogger(__name__)


class DetectTrackCountPipeline:
    """Pipeline for detection, tracking, and counting."""

    def __init__(
        self,
        detector_cfg: Optional[ObjectDetectionConfig] = None,
        tracker_cfg: Optional[TrackerConfig] = None,
    ):
        self.detector_cfg = detector_cfg if detector_cfg else ObjectDetectionConfig()
        self.tracker_cfg = tracker_cfg if tracker_cfg else TrackerConfig()
        self.nms_config = NMSConfig()

    @safe_execution(default_return=[])
    def _run(
        self,
        file: str,
        output_dir: str,
        export_types: Optional[List[ExportType]] = None,
    ) -> List:
        logging.info(f"Currently processing {file}")
        dataset_cfg = YOLODatasetConfig(filepath=file)
        detector = ObjectDetectionPipeline(self.detector_cfg, dataset_cfg)
        detections = detector()

        metadata = detector.metadata

        # Get low confidence for ByteTrack
        self.nms_config.conf = 0.1
        low_output = run_nms(
            detections.pred_bboxes,
            metadata.image_meter_width,
            detections.width,
            dataset_cfg.batch_size,
            self.nms_config,
        )

        # Get high confidence for ByteTrack
        self.nms_config.conf = 0.3
        high_output = run_nms(
            detections.pred_bboxes,
            metadata.image_meter_width,
            detections.width,
            dataset_cfg.batch_size,
            self.nms_config,
        )

        # Prepare bounding boxes for tracking pipeline
        low_preds, og_width, og_height = normalize_boxes_for_tracking(
            detections.image_shape,
            low_output,
            detections.width,
            detections.height,
            batch_size=dataset_cfg.batch_size,
        )
        high_preds, og_width, og_height = normalize_boxes_for_tracking(
            detections.image_shape,
            high_output,
            detections.width,
            detections.height,
            batch_size=dataset_cfg.batch_size,
        )

        tracker_output = run_tracker(
            low_preds,
            high_preds,
            metadata.image_meter_width,
            metadata.image_meter_height,
            self.tracker_cfg,
        )

        mot_tracks = tracker_output_to_mot(asdict(tracker_output))

        if export_types is None:
            export_types_list = []

        elif isinstance(export_types, ExportType):
            export_types_list = [export_types]

        else:
            export_types_list = export_types

        if ExportType.MOT in export_types_list:
            save_to_disk(mot_tracks, output_dir, export_types=ExportType.MOT)

        (left_count, right_count), crossing_frames = Count().count(mot_tracks)

        if crossing_frames:
            formatted_crossings = [
                {
                    "fish_id": track,
                    "direction": "left",
                    "frame_id": frame,
                    "file_name": Path(file).name,
                    "bbox": bbox,
                    "metadata": metadata,
                }
                for track, frame, bbox in crossing_frames["left"]
            ] + [
                {
                    "fish_id": track,
                    "direction": "right",
                    "frame_id": frame,
                    "file_name": Path(file).name,
                    "bbox": bbox,
                    "metadata": metadata,
                }
                for track, frame, bbox in crossing_frames["right"]
            ]
        else:
            formatted_crossings = []
            logger.info(f"No crossing frames detected for {file}")

        # Exclude MOT before saving crossings
        remaining_export_types = [
            et
            for et in export_types_list
            if et != ExportType.MOT and (et != ExportType.SUMMARY_CSV)
        ]
        save_to_disk(
            [formatted_crossings], output_dir, export_types=remaining_export_types
        )

        return formatted_crossings

    def run(
        self,
        file: List[str] | str,
        output_dir: str,
        export_types: Optional[List[ExportType]] = None,
    ) -> List[List[dict]] | List[dict]:
        """Run preprocessing, detection, tracking, and counting on frames.

        Args:
            file (List[str] | str): File(s) to process. Must be a path to an ARIS file or a directory holding ARIS files
            output_dir (str): Output directory to save results to

        Returns:
            dict: Tracking results and counts.
        """

        def is_valid_path(file_path: str) -> bool:
            # Check if it's a valid file and ends with '.aris' or '.ddf'
            return (
                os.path.exists(file_path)
                and os.path.isfile(file_path)
                and (file_path.endswith(".aris") or file_path.endswith(".ddf"))
            )

        def is_valid_directory(dir_path: str) -> bool:
            return os.path.isdir(dir_path) and any(
                f.endswith((".aris", ".ddf")) for f in os.listdir(dir_path)
            )

        if isinstance(file, str):
            if is_valid_path(file):
                return [self._run(file, output_dir, export_types)]

            elif is_valid_directory(file):
                # If path is a directory containing ARIS or DIDSON files, process all ARIS or DIDSON files in the
                # directory
                files = [
                    os.path.join(file, f)
                    for f in os.listdir(file)
                    if f.endswith((".aris", ".ddf"))
                ]

                return [self._run(f, output_dir, export_types) for f in files]

            else:
                raise ValueError(f"Invalid file or directory path: {file}")

        elif isinstance(file, list):
            valid_files = [f for f in file if is_valid_path(f) or is_valid_directory(f)]

            if len(valid_files) < len(file):
                logger.info(
                    f"Skipping invalid file path(s): {', '.join(set(file) - set(valid_files))}"
                )

            return [self._run(f, output_dir, export_types) for f in valid_files]

        else:
            raise ValueError(
                "Input should be a string or a list of strings representing file paths."
            )

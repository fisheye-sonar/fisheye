import os
from dataclasses import asdict
from typing import Optional, List

from fisheye.boxes import run_nms, normalize_boxes_for_tracking
from fisheye.configs import ObjectDetectionConfig, YOLODatasetConfig
from fisheye.configs.inference import TrackerConfig, NMSConfig
from fisheye.count.counter import Count
from fisheye.format import tracker_output_to_mot
from fisheye.pipelines import ObjectDetectionPipeline
from fisheye.track.tracker import run_tracker


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

    def _run(self, file: str):
        dataset_cfg = YOLODatasetConfig(filepath=file)
        detections = ObjectDetectionPipeline(self.detector_cfg, dataset_cfg).run()

        # Get low confidence for ByteTrack
        self.nms_config.conf = 0.1
        low_output = run_nms(
            detections.pred_bboxes,
            dataset_cfg.image_meter_width,
            detections.width,
            dataset_cfg.batch_size,
            self.nms_config,
        )

        # Get high confidence for ByteTrack
        self.nms_config.conf = 0.3
        high_output = run_nms(
            detections.pred_bboxes,
            dataset_cfg.image_meter_width,
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
            dataset_cfg.image_meter_width,
            dataset_cfg.image_meter_height,
            self.tracker_cfg,
        )

        mot_tracks = tracker_output_to_mot(asdict(tracker_output))
        (left_count, right_count), crossing_frames = Count().count(mot_tracks)

        return {
            "tracks": mot_tracks,
            "counts": (left_count, right_count),
            "file": file,
            "crossing_frames": crossing_frames,
        }

    def run(self, file: List[str] | str) -> List[dict] | dict:
        """Run preprocessing, detection, tracking, and counting on frames.

        Args:
            file (List[str] | str): File(s) to process. Must be a path to an ARIS file or a directory holding ARIS files

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
                return self._run(file)
            elif is_valid_directory(file):
                # If path is a directory containing ARIS or DIDSON files, process all ARIS or DIDSON files in the
                # directory
                files = [
                    os.path.join(file, f)
                    for f in os.listdir(file)
                    if f.endswith((".aris", ".ddf"))
                ]

                return [self._run(f) for f in files]

            else:
                raise ValueError(f"Invalid file or directory path: {file}")

        elif isinstance(file, list):
            valid_files = [f for f in file if is_valid_path(f) or is_valid_directory(f)]

            if len(valid_files) < len(file):
                print(
                    f"Skipping invalid file path(s): {', '.join(set(file) - set(valid_files))}"
                )

            return [self._run(f) for f in valid_files]

        else:
            raise ValueError(
                "Input should be a string or a list of strings representing file paths."
            )

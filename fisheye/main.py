from dataclasses import asdict

from fisheye.boxes import run_nms, normalize_boxes_for_tracking
from fisheye.configs import YOLODatasetConfig, YOLOv5ModelConfig, ObjectDetectionConfig
from fisheye.configs.inference import TrackerConfig, NMSConfig
from fisheye.count.counter import Count
from fisheye.format import tracker_output_to_mot
from fisheye.pipelines.detection import ObjectDetectionPipeline
from fisheye.track.tracker import run_tracker

weights = "/Users/madison/Code/fisheye-advanced/models/cfc_all_v5s_best.pt"
fp = "/Users/madison/Code/fisheye/tests/sample.aris"
# fp = '/Users/madison/Downloads/2024-11-02_2000_Van_Duzen_River_2000_2020.aris'

# Define configs here for inference
dataset_cfg = YOLODatasetConfig(filepath=fp)
model_cfg = YOLOv5ModelConfig(weights=weights)
detection_cfg = ObjectDetectionConfig(model=model_cfg)
tracking_config = TrackerConfig()
nms_config = NMSConfig()

detection_results = ObjectDetectionPipeline(detection_cfg, dataset_cfg).run()

# Get low confidence for ByteTrack
nms_config.conf = 0.1
low_output = run_nms(
    detection_results.pred_bboxes,
    dataset_cfg.image_meter_width,
    detection_results.width,
    dataset_cfg.batch_size,
    nms_config,
)

# Get high confidence for ByteTrack
nms_config.conf = 0.3
high_output = run_nms(
    detection_results.pred_bboxes,
    dataset_cfg.image_meter_width,
    detection_results.width,
    dataset_cfg.batch_size,
    nms_config,
)

# Prepare bounding boxes for tracking pipeline
low_preds, og_width, og_height = normalize_boxes_for_tracking(
    detection_results.image_shape,
    low_output,
    detection_results.width,
    detection_results.height,
    batch_size=dataset_cfg.batch_size,
)
high_preds, og_width, og_height = normalize_boxes_for_tracking(
    detection_results.image_shape,
    high_output,
    detection_results.width,
    detection_results.height,
    batch_size=dataset_cfg.batch_size,
)

tracker_output = run_tracker(
    low_preds,
    high_preds,
    dataset_cfg.image_meter_width,
    dataset_cfg.image_meter_height,
    tracking_config,
)

mot_tracks = tracker_output_to_mot(asdict(tracker_output))
left_count, right_count = Count(protocol="LOI").count(mot_tracks)

print(f"Left: {left_count}, Right: {right_count}")


if __name__ == "__main__":
    pass

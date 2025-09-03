from fisheye.detect.yolov11 import YOLOv11ObjectDetectionModel
from fisheye.detect.yolov5 import YOLOv5ObjectDetectionModel
from fisheye.enums import DetectorType


DETECTOR_CLASS_REGISTRY = {
    DetectorType.YOLOv5: YOLOv5ObjectDetectionModel,
    DetectorType.YOLOv11: YOLOv11ObjectDetectionModel,
}

from dataclasses import dataclass

import torch
import yolov5

from fisheye.models.base import BaseModel


@dataclass
class YOLOv5ModelConfig(BaseModel):
    """YOLOv5 model config.

    Exposing class variables from YOLOv5's AutoShape.
    """

    conf: float = 0.05  # NMS confidence threshold
    iou: float = 0.2  # NMS IoU threshold
    agnostic: bool = False  # NMS class-agnostic
    multi_label: bool = False  # NMS multiple labels per box
    classes: list[int] | None = (
        None  # (Optional list) filter by class, i.e. = [0, 15, 16] for COCO
    )
    max_det: int = 1000  # Maximum number of detections per image
    amp: bool = False  # Automatic Mixed Precision (AMP) inference


class YOLOv5ObjectDetectionModel(BaseModel):
    """
    YOLOv5 object detection model class for inference.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        config: YOLOv5ModelConfig = YOLOv5ModelConfig(),
    ) -> None:
        """Initializes the YOLOv5 model by loading weights and setting the device.

        Args:
            model_path (Union[str, Path]): Local path to the YOLO model weights.
            device (torch.device): The device (CPU or GPU) to run inference on.
        """
        self.config = config
        super().__init__(model_path, device)

    def _load_model(self, weights, device):
        """Loads the weights & device. Modified version from Ultralytics."""
        model = yolov5.load(weights, device)
        # Set model parameters
        model.conf = self.config.conf
        model.iou = self.config.iou
        model.agnostic = self.config.agnostic
        model.multi_label = self.config.multi_label
        model.classes = self.config.classes
        model.max_det = self.config.max_det
        model.amp = self.config.amp

        return model

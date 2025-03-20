import torch
import yolov5
import warnings

from fisheye.configs import YOLOv5ModelConfig
from fisheye.detect.base import BaseModel


# Suppress the `torch.cuda.amp.autocast(args...)` warning is deprecated.* raised in yolov5.models.common
warnings.filterwarnings(
    "ignore",
    message=".*autocast.*",  # Use regex to match part of the message
    category=FutureWarning,
    module="yolov5.models.common",
)


class YOLOv5ObjectDetectionModel(BaseModel):
    """
    YOLOv5 object detection model class for inference.
    """

    def __init__(
        self,
        config: YOLOv5ModelConfig = YOLOv5ModelConfig(),
    ) -> None:
        """Initializes the YOLOv5 model by loading weights and setting the device.

        Args:
            model_path (Union[str, Path]): Local path to the YOLO model weights.
            device (torch.device): The device (CPU or GPU) to run inference on.
        """
        self.config = config
        super().__init__(self.config.weights, self.config.device)

    def _load_model(self, weights, device):
        """Loads the weights & device. Modified version from Ultralytics."""
        model = yolov5.load(weights, device)

        # Set model parameters
        model.agnostic = self.config.agnostic
        model.multi_label = self.config.multi_label
        model.classes = self.config.classes
        model.max_det = self.config.max_det
        model.amp = self.config.amp

        return model

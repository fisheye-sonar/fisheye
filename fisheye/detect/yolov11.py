from typing import Union, List

import torch
from ultralytics import YOLO

from fisheye.configs import YOLOv11ModelConfig
from fisheye.detect.base import BaseModel


class YOLOv11ObjectDetectionModel(BaseModel):
    """
    YOLOv11 object detection model class for inference.
    """

    def __init__(
        self,
        config: YOLOv11ModelConfig = YOLOv11ModelConfig(),
    ) -> None:
        """Initializes the YOLOv11 model by loading weights and setting the device.

        Args:
            model_path (Union[str, Path]): Local path to the YOLO model weights.
            device (torch.device): The device (CPU or GPU) to run inference on.
        """
        self.config = config
        super().__init__(self.config.weights, self.config.device)

    def _load_model(self, weights, device):
        """Loads the weights & device."""
        model = YOLO(weights).model

        return model

    @torch.inference_mode()
    def predict(self, images: Union[torch.Tensor, List]):
        """Forward pass for the model."""
        prediction, _ = self.model(images)

        # e.g. shape(1, 84, 6300) to shape(1, 6300, 84)
        prediction = prediction.transpose(-1, -2)

        # Single-class - default to 1 so that it doesn't affect NMS
        cls_idx = torch.ones(
            (*prediction.shape[:2], 1),  # [B, N, 1]
            device=prediction.device,
        )

        # Concatenate [x1, y1, x2, y2, conf] + cls_idx
        # [B, N, 6]
        output = torch.cat([prediction, cls_idx], dim=2)

        return output

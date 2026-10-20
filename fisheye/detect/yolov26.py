from typing import Union, List

import torch

from fisheye.configs import YOLOv26ModelConfig
from fisheye.detect.base import BaseModel


class YOLOv26ObjectDetectionModel(BaseModel):
    """
    YOLOv26 object detection model class for inference.
    """

    def __init__(
        self,
        config: YOLOv26ModelConfig = YOLOv26ModelConfig(),
    ) -> None:
        """Initializes the YOLOv26 model by loading weights and setting the device.

        Args:
            config (YOLOv26ModelConfig): Model configuration containing weights path and device.
        """
        self.config = config
        super().__init__(self.config.weights, self.config.device)

    def _load_model(self, weights, device):
        """Loads the weights & device."""
        from ultralytics import YOLO

        model = YOLO(weights).model

        return model

    def predict(self, images: Union[torch.Tensor, List]):
        """Forward pass for the model."""
        # returns [B, 5, N] - [x1, y1, x2, y2, conf]
        prediction, _ = self.model(images)

        # e.g. shape(1, 84, 6300) to shape(1, 6300, 84)
        prediction = prediction.transpose(-1, -2)

        # Concatenate a single-class column (1) directly so that it doesn't affect NMS + filtering
        # output.shape = [B, N, 6]
        output = torch.cat(
            [
                prediction,
                torch.ones((*prediction.shape[:2], 1), device=prediction.device),
            ],
            dim=2,
        )

        return output

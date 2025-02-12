from dataclasses import dataclass

import torch

from fisheye.config import YOLODatasetConfig
from fisheye.dataloaders.yolo import create_yolo_dataloader
from fisheye.models.base import BaseModel


@dataclass
class ObjectDetectionOutput:
    """Object Detection Output."""

    pred_bboxes: torch.Tensor = None
    image_shape: torch.Tensor = None
    width: int = None
    height: int = None


class ObjectDetectionPipeline:
    """Detection pipeline class

    Orchestrates the entire detection pipeline.
    """

    def __init__(
        self,
        model: BaseModel,
        device: torch.device,
        config: YOLODatasetConfig = YOLODatasetConfig(),
        do_suppression: bool = True,
    ):
        self.device = device
        self.model = model
        self.dataloader, self.dataset = create_yolo_dataloader(config)

    def __call__(self, *args, **kwargs):
        """Executes the detection pipeline."""
        return self.run()

    def preprocess(self, image):
        image = image.to(self.device, non_blocking=True)
        image = (
            image.half() if self.device.type != "cpu" else image.float()
        )  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0

        return image

    def postprocess(self, model_outputs):
        """Process model outputs."""

        processed_outputs = [
            {"detections": output, "shape": shape}
            for output, shape in zip(model_outputs, image_shapes)
        ]
        return processed_outputs

    def _forward(self):
        """Performs inference.

        Returns:
            Tuple[List[Any], List[Tuple]]:
                - Model output(s)
                - Image shapes for resizing predictions to original dimensions
        """
        inference = []
        image_shapes = []
        for batch_idx, (img, _, shapes) in enumerate(self.dataloader):
            img = self.preprocess(img)
            size = tuple(img.shape)
            nb, _, height, width = size  # batch size, channels, height, width
            inf_out, _ = self.model.predict(img)

            # Save shapes for resizing to original shape
            batch_shape = []
            for si, pred in enumerate(inf_out):
                batch_shape.append((img[si].shape[1:], shapes[si]))

            image_shapes.append(batch_shape)
            inference.append(inf_out)

        return inference, image_shapes, width, height

    def run(self):
        """Executes the detection pipeline end-to-end.

        Returns:
            List[Any]: Processed detection results.
        """
        raw_outputs, shapes = self._forward()
        processed_results = self.postprocess(raw_outputs, shapes)

        return processed_results

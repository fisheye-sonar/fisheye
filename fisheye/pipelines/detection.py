from functools import partial
from typing import Dict, Any, Optional

from fisheye.configs import (
    YOLODatasetConfig,
    ObjectDetectionPipelineOutput,
    ObjectDetectionConfig,
)
from fisheye.dataloaders.yolo import create_yolo_dataloader
from fisheye.models.yolov5 import YOLOv5ObjectDetectionModel


class ObjectDetectionPipeline:
    """Detection pipeline class

    Orchestrates the entire detection pipeline.
    """

    def __init__(
        self,
        config: ObjectDetectionConfig = ObjectDetectionConfig(),
        dataset_config: Optional[YOLODatasetConfig] = None,
        postprocessing_params: Dict[str, Any] = None,
        *args,
        **kwargs,
    ):
        model = config.model
        self.device = model.device

        if not model:
            raise ValueError("A model must be specified in the pipeline configuration.")

        self.model = (
            YOLOv5ObjectDetectionModel(model)
            if isinstance(model.weights, str)
            else model.weights
        )

        if dataset_config is None:
            dataset_config = YOLODatasetConfig(*args, **kwargs)

        self.dataloader, self.dataset = create_yolo_dataloader(dataset_config)
        self.postprocessing_params = self._sanitize_postprocessing_params(
            postprocessing_params
        )

    def _sanitize_postprocessing_params(self, postprocessing):
        """Converts postprocessing dict into a list of callable functions with parameters."""
        if not postprocessing:
            return postprocessing

        configured_steps = []
        for step, params in postprocessing.items():
            if params:
                configured_steps.append(partial(step, **params))  # Bind parameters
            else:
                configured_steps.append(step)

        return configured_steps

    def __call__(self, *args, **kwargs):
        """Executes the detection pipeline."""
        output = self.run()
        processed_output = self.postprocess(output)

        return processed_output

    def preprocess(self, image):
        image = image.to(self.device, non_blocking=True)
        image = (
            image.half() if self.device != "cpu" else image.float()
        )  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0

        return image

    def postprocess(self, output: ObjectDetectionPipelineOutput):
        """Process model output sequentially."""

        processed_output = output
        if self.postprocessing_params:
            for step in self.postprocessing_params:
                step_params = {
                    k: v for k, v in output.__dict__.items() if k in step.keywords
                }
                if "image_pixel_width" not in step_params and hasattr(output, "width"):
                    step_params["image_pixel_width"] = output.width

                processed_output = step(processed_output, **step_params)

        return processed_output

    def _forward(self, *args, **kwargs):
        """Performs inference.

        Returns:
            Tuple[List[Any], List[Tuple]]:
                - Model output(s)
                - Image shapes for resizing predictions to original dimensions
        """
        inference = []
        image_shapes = []
        width = None
        height = None

        for batch_idx, (img, _, shapes) in enumerate(self.dataloader):
            img = self.preprocess(img)
            size = tuple(img.shape)
            nb, _, height, width = size  # batch size, channels, height, width
            inf_out = self.model.predict(img)

            # Save shapes for resizing to original shape
            batch_shape = []
            for si, pred in enumerate(inf_out):
                batch_shape.append((img[si].shape[1:], shapes[si]))

            image_shapes.append(batch_shape)
            inference.append(inf_out)

        return inference, image_shapes, width, height

    def run(self, *args, **kwargs):
        """Executes the detection pipeline end-to-end.

        Returns:
            List[Any]: Processed detection results.
        """
        return ObjectDetectionPipelineOutput(*self._forward(*args, **kwargs))

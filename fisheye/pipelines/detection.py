from functools import partial
from typing import Dict, Any, Optional

from fisheye.configs import (
    YOLODatasetConfig,
    ObjectDetectionPipelineOutput,
    ObjectDetectionConfig,
)

from fisheye.boxes import run_nms
from fisheye.dataloaders.yolo import create_yolo_dataloader
from fisheye.detect.yolov5 import YOLOv5ObjectDetectionModel


# Add postprocessing methods to this registry
POSTPROCESSING_REGISTRY = {
    "nms": run_nms,
}


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
        self.postprocessing_steps = (
            self._build_postprocessing_params(postprocessing_params)
            if postprocessing_params
            else postprocessing_params
        )

    def _build_postprocessing_params(self, postprocessing_params):
        """Sanitizes postprocessing parameters to format them correctly."""
        if not postprocessing_params:
            return []
        postprocessing_steps = []

        for step_name, params in postprocessing_params.items():
            processor = POSTPROCESSING_REGISTRY.get(step_name)
            if not processor:
                raise ValueError(f"Unknown postprocessing step: {step_name}")

            if isinstance(params, list):
                for p in params:
                    postprocessing_steps.append(partial(processor, nms_config=p))

            elif isinstance(params, dict):
                for p in params:
                    postprocessing_steps.append(partial(processor, nms_config=[p]))

            else:
                postprocessing_steps.append(partial(processor, **params))

        return postprocessing_steps

    def __call__(self, *args, **kwargs):
        """Executes the detection pipeline."""
        return self.run(*args, **kwargs)

    def preprocess(self, image):
        image = image.to(self.device, non_blocking=True)
        image = (
            image.half() if self.device != "cpu" else image.float()
        )  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0

        return image

    def postprocess(self, output):
        """Process model output using configured postprocessing steps."""

        processed_output = []
        for step in self.postprocessing_steps:
            step.keywords["pred_bboxes"] = output.pred_bboxes
            step.keywords["image_meter_width"] = self.dataset.image_meter_width
            step.keywords["image_pixel_width"] = output.width
            step.keywords["batch_size"] = self.dataset.batch_size

            # Append the result of each step to the list
            processed_output.append(step(**step.keywords))

        return processed_output if processed_output else output

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
        output = ObjectDetectionPipelineOutput(*self._forward(*args, **kwargs))

        return output if not self.postprocessing_steps else self.postprocess(output)

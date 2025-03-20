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
        self.postprocessing_steps = self._sanitize_postprocessing_params(
            postprocessing_params
        )

    def _sanitize_postprocessing_params(self, postprocessing_params):
        """Sanitizes postprocessing parameters to format them correctly."""
        if not postprocessing_params:
            return []
        postprocessing_steps = []

        for step_name, params in postprocessing_params.items():
            if isinstance(params, list):  # Case where multiple NMS calls are expected
                for step_params in params:
                    postprocessing_steps.append(
                        partial(globals()[step_name], **step_params)
                    )
            elif isinstance(params, dict):  # Standard case with one set of params
                if not any(
                    step.func == globals()[step_name] for step in postprocessing_steps
                ):
                    postprocessing_steps.append(partial(globals()[step_name], **params))
            else:
                postprocessing_steps.append(globals()[step_name])

        return postprocessing_steps

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

    def postprocess(self, output, save_sequentially=False):
        """Process model output using configured postprocessing steps.

        output: ObjectDetectionPipelineOutput
        save_sequentially (bool): Save postprocessing output sequentially or separately.
        """

        processed_output = output

        if save_sequentially:
            # Apply each step sequentially and modify the output progressively
            for step in self.postprocessing_steps:
                step.keywords["pred_bboxes"] = processed_output.pred_bboxes
                step.keywords["image_meter_width"] = self.dataset.image_meter_width
                step.keywords["image_pixel_width"] = processed_output.width
                step.keywords["batch_size"] = self.dataset.batch_size

                # Apply the step and update the processed_output
                processed_output = step(**step.keywords)

        else:
            # Save results separately
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
        return ObjectDetectionPipelineOutput(*self._forward(*args, **kwargs))

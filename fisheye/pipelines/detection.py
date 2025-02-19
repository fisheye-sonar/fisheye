from fisheye.dataclasses import (
    YOLODatasetConfig,
    YOLOv5ModelConfig,
    ObjectDetectionPipelineOutput,
)
from fisheye.dataloaders.yolo import create_yolo_dataloader
from fisheye.models.yolov5 import YOLOv5ObjectDetectionModel


class ObjectDetectionPipeline:
    """Detection pipeline class

    Orchestrates the entire detection pipeline.
    """

    def __init__(
        self,
        pipeline_cfg: YOLOv5ModelConfig = YOLOv5ModelConfig(),
        dataset_cfg: YOLODatasetConfig = YOLODatasetConfig(),
    ):
        self.device = pipeline_cfg.device
        if not pipeline_cfg.model:
            raise ValueError("A model must be specified in the pipeline configuration.")

        self.model = (
            YOLOv5ObjectDetectionModel(pipeline_cfg)
            if isinstance(pipeline_cfg.model, str)
            else (pipeline_cfg.model)
        )

        self.dataloader, self.dataset = create_yolo_dataloader(dataset_cfg)

    def __call__(self, *args, **kwargs):
        """Executes the detection pipeline."""
        return self.run()

    def preprocess(self, image):
        image = image.to(self.device, non_blocking=True)
        image = (
            image.half() if self.device != "cpu" else image.float()
        )  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0

        return image

    # def postprocess(self, model_outputs):
    #     """Process model outputs."""
    #
    #     processed_outputs = [
    #         {"detections": output, "shape": shape}
    #         for output, shape in zip(model_outputs, image_shapes)
    #     ]
    #     return processed_outputs

    def predict(self):
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
            inf_out = self.model(img)

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
        return ObjectDetectionPipelineOutput(*self.predict())

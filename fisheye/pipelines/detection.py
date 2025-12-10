from functools import partial
from typing import Dict, Any, Optional

import structlog
import torch

from fisheye.boxes import run_nms
from fisheye.common.generic import run_with_threads
from fisheye.common.logging import log_progress
from fisheye.configs import (
    YOLODatasetConfig,
    ObjectDetectionPipelineOutput,
    ObjectDetectionConfig,
)

from fisheye.dataloaders import create_dataloader
from fisheye.detect.base import BaseModel
from tqdm import tqdm
from fisheye.lengths.length_estimator import LengthEstimator

from fisheye.boxes import run_nms, normalize_boxes_for_tracking

# Add postprocessing methods to this registry
POSTPROCESSING_REGISTRY = {
    "nms": run_nms,
}

logger = structlog.get_logger()


class ObjectDetectionPipeline:
    """Detection pipeline class

    Orchestrates the entire detection pipeline.
    """

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        config: ObjectDetectionConfig = ObjectDetectionConfig(),
        postprocessing_params: Dict[str, Any] = None,
        *args,
        **kwargs,
    ):
        if not model:
            raise ValueError("A model must be specified in the pipeline configuration.")

        self.model = model
        self.device = model.config.device

        logger.info(
            f"initialized_detector", model=type(self.model).__name__, device=self.device
        )

        self.dataloader: Optional[YOLODatasetConfig] = None
        self.dataset: Optional[Any] = None
        self.metadata: Optional[Any] = None

        self.use_multithreading = config.use_multithreading
        self.apply_nms_batchwise = config.apply_nms_batchwise
        self.apply_length_estimates_batchwise = config.apply_length_estimates_batchwise
        self.nms_config = config.nms_config

        self.max_workers = config.max_workers
        self.postprocessing_steps = (
            self._build_postprocessing_params(postprocessing_params)
            if postprocessing_params
            else postprocessing_params
        )

        self.length_estimator = None

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
                    if p:
                        postprocessing_steps.append(partial(processor, nms_config=p))

            else:
                postprocessing_steps.append(partial(processor, nms_config=params))

        return postprocessing_steps

    def __call__(self, *args, **kwargs):
        """Executes the detection pipeline."""
        return self.run(*args, **kwargs)

    def load_dataset(self, dataset_config: YOLODatasetConfig):
        """Initializes the dataloader + dataset for the pipeline."""
        self.dataloader, self.dataset = create_dataloader(dataset_config)
        self.metadata = self.dataset.metadata

        return self.dataloader, self.dataset

    def preprocess(self, image):
        image = image.to(self.device, non_blocking=True)
        image = (
            image.half() if self.device != "cpu" else image.float()
        )  # uint8 to fp16/32
        # image /= 255.0  # 0 - 255 to 0.0 - 1.0 # MAH 2025-12-08 11:01:06 now done in postprocessing

        return image

    def postprocess(self, output):
        """Process model output using configured postprocessing steps."""

        processed_output = []
        for step in self.postprocessing_steps:
            step.keywords["pred_bboxes"] = output.pred_bboxes
            step.keywords["image_meter_width"] = self.metadata.image_meter_width
            step.keywords["image_pixel_width"] = output.width
            step.keywords["batch_size"] = self.dataset.batch_size

            # Append the result of each step to the list
            processed_output.append(step(**step.keywords))

        return processed_output if processed_output else output

    def _forward(self, *args, **kwargs):
        """Performs inference with optional multithreading."""

        if self.length_estimator is None:
            if self.metadata is None:
                # raise ValueError("Metadata is not set cant apply length estimates")
                pass
            else:
                self.length_estimator = LengthEstimator(self.metadata)

        if self.apply_nms_batchwise:
            all_low_preds_updated_batch = {}
            all_high_preds_updated_batch = {}
            all_low_length_estimates = {}
            all_high_length_estimates = {}
        else:
            inference_bboxes = []
            image_shapes = []
            width = None
            height = None

        with torch.inference_mode():
            for batch_idx, (img, _, shapes, img_original) in tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                desc="Running detection",
            ):
                img = self.preprocess(img)
                size = tuple(img.shape)
                nb, _, height, width = size  # batch size, channels, height, width

                # Save shapes for resizing to original shape
                batch_shape = []
                for si in range(img.shape[0]):
                    batch_shape.append((img[si].shape[1:], shapes[si]))

                if self.use_multithreading:
                    # per image inference with multithreading
                    img_list = [img[i : i + 1] for i in range(img.shape[0])]
                    inf_out = run_with_threads(self.model, img_list, self.max_workers)
                    # Concatenate per sample predictions into a batched tensor [B, N, 6]
                    inf_out = torch.cat(inf_out, dim=0)
                else:
                    # Batched inference - [B, N, 6]
                    inf_out = self.model(img)

                del img
                torch.cuda.empty_cache()

                if self.apply_nms_batchwise:
                    # Get low confidence for ByteTrack
                    self.nms_config.conf = 0.1
                    low_output = run_nms(
                        [inf_out.cpu()],  # xyxy format relative to YOLO pixel space
                        self.metadata.image_meter_width,
                        width,
                        self.dataset.batch_size,
                        self.nms_config,
                    )

                    # Get high confidence for ByteTrack
                    self.nms_config.conf = 0.3
                    high_output = run_nms(
                        [inf_out.cpu()],  # xyxy format relative to YOLO pixel space
                        self.metadata.image_meter_width,
                        width,
                        self.dataset.batch_size,
                        self.nms_config,
                    )

                    # Prepare bounding boxes for tracking pipeline
                    low_preds, _og_width, _og_height = normalize_boxes_for_tracking(
                        [batch_shape],
                        low_output,  # xyxy format relative to YOLO pixel space
                        width,
                        height,
                        batch_size=self.dataset.batch_size,
                    )
                    high_preds, _og_width, _og_height = normalize_boxes_for_tracking(
                        [batch_shape],
                        high_output,  # xyxy format relative to YOLO pixel space
                        width,
                        height,
                        batch_size=self.dataset.batch_size,
                    )

                    all_low_preds_updated_batch.update(
                        {(batch_idx, k[1]): v for k, v in low_preds.items()}
                    )

                    all_high_preds_updated_batch.update(
                        {(batch_idx, k[1]): v for k, v in high_preds.items()}
                    )

                    if self.apply_length_estimates_batchwise:
                        img_original = self.preprocess(img_original)

                        low_length_estimates = self.length_estimator.run(
                            img_original, low_preds
                        )
                        low_length_estimates = {
                            (batch_idx, k): v for k, v in low_length_estimates.items()
                        }

                        all_low_length_estimates.update(low_length_estimates)

                        high_length_estimates = self.length_estimator.run(
                            img_original, high_preds
                        )
                        high_length_estimates = {
                            (batch_idx, k): v for k, v in high_length_estimates.items()
                        }
                        all_high_length_estimates.update(high_length_estimates)

                        del img_original
                        torch.cuda.empty_cache()
                else:
                    image_shapes.append(batch_shape)
                    inference_bboxes.append(inf_out.cpu())

                log_progress(
                    logger,
                    batch_idx,
                    len(self.dataloader),
                    prefix="Detector progress | ",
                )

        if self.apply_nms_batchwise:
            if self.apply_length_estimates_batchwise:
                return (
                    all_low_preds_updated_batch,
                    all_high_preds_updated_batch,
                    all_low_length_estimates,
                    all_high_length_estimates,
                )
            else:
                return all_low_preds_updated_batch, all_high_preds_updated_batch

        else:
            return inference_bboxes, image_shapes, width, height

    def run(self, *args, **kwargs):
        """Executes the detection pipeline end-to-end.

        Returns:
            List[Any]: Processed detection results.
        """
        if self.apply_nms_batchwise:
            return self._forward(*args, **kwargs)

        else:
            output = ObjectDetectionPipelineOutput(*self._forward(*args, **kwargs))

            return output if not self.postprocessing_steps else self.postprocess(output)

from typing import Dict, Any, Optional, Tuple

import structlog
import torch

from fisheye.boxes import NMSProcessor
from fisheye.common.generic import run_with_threads
from fisheye.common.logging import log_progress
from fisheye.configs import (
    YOLODatasetConfig,
    ObjectDetectionConfig,
)
from fisheye.configs.inference import NMSConfig
from fisheye.dataloaders import create_dataloader
from fisheye.detect.base import BaseModel

logger = structlog.get_logger()


class ObjectDetectionPipeline:
    """Detection pipeline class

    Orchestrates the entire detection pipeline.
    """

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        config: ObjectDetectionConfig = ObjectDetectionConfig(),
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
        self.max_workers = config.max_workers
        self.nms_config = NMSConfig()
        self.apply_nms_batchwise = config.apply_nms_batchwise

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
        image /= 255.0  # 0 - 255 to 0.0 - 1.0

        return image

    def _run_inference(self, *args, **kwargs):
        """Performs inference with optional multithreading."""

        nms_processor = None
        all_low_preds, all_high_preds = {}, {}

        if self.apply_nms_batchwise:
            nms_processor = NMSProcessor(
                self.nms_config, self.metadata, self.dataset.batch_size
            )

        with torch.inference_mode():
            for batch_idx, (img, _, shapes, original_img) in enumerate(self.dataloader):
                img = self.preprocess(img)
                size = tuple(img.shape)
                nb, _, height, width = size  # batch size, channels, height, width

                if self.use_multithreading:
                    # per image inference with multithreading
                    img_list = [img[i : i + 1] for i in range(img.shape[0])]
                    batch_out = run_with_threads(self.model, img_list, self.max_workers)
                    # Concatenate per sample predictions into a batched tensor [B, N, 6]
                    batch_out = torch.cat(batch_out, dim=0)
                else:
                    # Batched inference - [B, N, 6]
                    batch_out = self.model(img)

                torch.cuda.empty_cache()

                # Save shapes for resizing to original shape
                batch_shape = []
                for si, pred in enumerate(batch_out):
                    batch_shape.append((img[si].shape[1:], shapes[si]))

                if nms_processor:
                    low_preds, high_preds = nms_processor.run(
                        batch_out.cpu(), batch_shape
                    )

                    all_low_preds.update(
                        {
                            (batch_idx, k[1]): v
                            for i, (k, v) in enumerate(low_preds.items())
                        }
                    )
                    all_high_preds.update(
                        {
                            (batch_idx, k[1]): v
                            for i, (k, v) in enumerate(high_preds.items())
                        }
                    )

                log_progress(
                    logger,
                    batch_idx,
                    len(self.dataloader),
                    prefix="Detector progress | ",
                )

        return all_low_preds, all_high_preds

    def run(self, *args, **kwargs) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
        """Executes the detection pipeline end-to-end.

        Returns:
            Tuple[Dict, Dict]: (low_preds, high_preds) dictionaries from batchwise NMS.
        """
        low_preds, high_preds = self._run_inference(*args, **kwargs)

        return low_preds, high_preds

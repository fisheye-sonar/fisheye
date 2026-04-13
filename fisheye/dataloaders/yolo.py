import numpy as np
import structlog

from fisheye.configs import YOLODatasetConfig
from fisheye.dataloaders import ARISBatchedDataset
from fisheye.dataloaders.yolo_mixin import YOLOPostprocessMixin

logger = structlog.get_logger()


class YOLOARISBatchedDataset(YOLOPostprocessMixin, ARISBatchedDataset):
    """YOLOARISBatchedDataset

    A PyTorch Dataset for loading ARIS/DIDSON data specifically tailored for YOLOv5-style object detection tasks.
    """

    def __init__(self, config: YOLODatasetConfig):
        super().__init__(config)

        self.stride = config.stride
        self.pad = config.pad
        self.img_size = config.img_size
        self.original_shape = (self.metadata.ydim, self.metadata.xdim)
        self.shape = self._compute_resized_shape()

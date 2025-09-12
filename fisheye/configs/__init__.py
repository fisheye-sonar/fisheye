from .datasets import BaseDatasetConfig, YOLODatasetConfig
from .inference import ObjectDetectionConfig, ObjectDetectionPipelineOutput

from .models import (
    BaseModelConfig,
    YOLOv5ModelConfig,
    YOLOv11ModelConfig,
    get_detector_config,
)

__all__ = [
    "BaseModelConfig",
    "YOLODatasetConfig",
    "YOLOv5ModelConfig",
    "YOLOv11ModelConfig",
    "get_detector_config",
    "ObjectDetectionConfig",
    "ObjectDetectionPipelineOutput",
]

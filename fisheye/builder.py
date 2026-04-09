from pathlib import Path
from typing import Union

from fisheye.configs import (
    ObjectDetectionConfig,
    YOLODatasetConfig,
    get_detector_config,
    get_length_model_config,
)
from fisheye.configs.datasets import ImageDatasetConfig
from fisheye.detect.factory import DETECTOR_CLASS_REGISTRY
from fisheye.enums import DetectorType, LengthEstimatorType
from fisheye.pipelines import ObjectDetectionPipeline
from fisheye.pipelines.pipeline import DetectTrackCountPipeline
from omegaconf import DictConfig


from fisheye.configs.inference import TargetSizeConfig


class PipelineFactory:
    """Builder class for constructing pipeline components."""

    @staticmethod
    def build_detector(platform_cfg: DictConfig, project_root: Path):
        """Build the object detector model."""
        model_config = platform_cfg.model
        weights = model_config.weights
        resolved_weights_path = str((project_root / weights).resolve())

        detector_type = DetectorType(model_config.type)
        detector_cfg = get_detector_config(detector_type, **model_config)
        detector_cfg.weights = resolved_weights_path

        return (
            DETECTOR_CLASS_REGISTRY[detector_type](detector_cfg),
            resolved_weights_path,
            detector_cfg,
        )

    @staticmethod
    def build_dataset_config(
        dataset_config: DictConfig,
    ) -> Union[YOLODatasetConfig, ImageDatasetConfig]:
        """Build the dataset configuration."""
        if "image_folder" in dataset_config:
            return ImageDatasetConfig(**dataset_config)

        return YOLODatasetConfig(**dataset_config)

    @staticmethod
    def build_runtime_config(
        platform_cfg: DictConfig, project_root: Path, detector_cfg
    ) -> dict:
        """Build the runtime configuration including length estimation."""
        runtime_config = dict(platform_cfg.inference)

        if "length_config" in runtime_config:
            length_cfg_dict = dict(runtime_config["length_config"])
            if length_cfg_dict.get("weights"):
                length_cfg_dict["weights"] = (
                    project_root / length_cfg_dict["weights"]
                ).resolve()

            model_type = length_cfg_dict.pop("type", LengthEstimatorType.UNET.value)
            runtime_config["length_config"] = get_length_model_config(
                model_type, **length_cfg_dict
            )

        runtime_config["model"] = detector_cfg
        return runtime_config

    @staticmethod
    def build_pipeline(
        detector,
        runtime_config: dict,
        dataset_cfg: YOLODatasetConfig,
        target_size: TargetSizeConfig = None,
    ) -> DetectTrackCountPipeline:
        """Build the main processing pipeline."""

        if runtime_config.get("apply_length_estimates_batchwise", False):
            dataset_cfg.return_original_image = True

        # Build out individual pipeline(s)
        detector_pipe = ObjectDetectionPipeline(
            model=detector, config=ObjectDetectionConfig(**runtime_config)
        )

        return DetectTrackCountPipeline(
            detector_pipe, dataset_cfg=dataset_cfg, target_size=target_size
        )

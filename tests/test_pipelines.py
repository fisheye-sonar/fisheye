from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from conftest import ARIS_FILE
from fisheye.configs import (
    ObjectDetectionConfig,
    YOLODatasetConfig,
    YOLOv5ModelConfig,
)
from fisheye.configs.inference import NMSConfig
from fisheye.detect.yolov5 import YOLOv5ObjectDetectionModel
from fisheye.pipelines import ObjectDetectionPipeline


def test_preprocess():
    """Test image preprocessing."""

    dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
    model_cfg = YOLOv5ModelConfig(weights="dummy/path")
    config = ObjectDetectionConfig(model=model_cfg)

    # Mock the model to be used in the pipeline
    mock_model = MagicMock()
    mock_predict_return = torch.rand((1, 6, 6))
    mock_model.predict.return_value = mock_predict_return

    # Patch the _load_model method to return the mocked model
    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):

        pipeline = ObjectDetectionPipeline(config, dataset_cfg)

        mock_image = torch.rand((1, 3, 640, 640))
        preprocessed_image = pipeline.preprocess(mock_image)

        assert preprocessed_image.device.type == model_cfg.device
        assert preprocessed_image.shape == (1, 3, 640, 640)
        assert (preprocessed_image <= 1.0).all()  # Ensure the image is normalized


def test_object_detection_pipeline_no_postprocessing():
    """Test ObjectDetectionPipeline with a mocked `self.model` (patched _load_model)."""

    # Mock the model to be used in the pipeline
    mock_model = MagicMock()
    mock_predict_return = torch.rand((1, 6, 6))
    mock_model.predict.return_value = mock_predict_return

    # Patch the _load_model method to return the mocked model
    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(model=model_cfg)
        batch_size = dataset_cfg.batch_size

        pipeline = ObjectDetectionPipeline(config, dataset_cfg)
        # Minus 1 for optical flow
        batch_size = (
            pipeline.metadata.numframes - 1
            if pipeline.metadata.numframes < batch_size
            else batch_size
        )
        pipeline.model = mock_model
        output = pipeline()

        assert mock_model.predict.call_count == batch_size
        assert len(output.pred_bboxes) == 1
        assert isinstance(output.pred_bboxes[0][0], torch.Tensor)
        assert output.pred_bboxes[0].shape == (
            batch_size,
            6,
            6,
        )  # Check the shape of the predictions
        assert output.image_shape == [
            [
                (
                    torch.Size([960, 512]),
                    (
                        (2686, 1307),
                        (
                            (0.33358153387937456, 0.33282325937260904),
                            (np.float64(38.5), np.float64(32.0)),
                        ),
                    ),
                ),
                (
                    torch.Size([960, 512]),
                    (
                        (2686, 1307),
                        (
                            (0.33358153387937456, 0.33282325937260904),
                            (np.float64(38.5), np.float64(32.0)),
                        ),
                    ),
                ),
                (
                    torch.Size([960, 512]),
                    (
                        (2686, 1307),
                        (
                            (0.33358153387937456, 0.33282325937260904),
                            (np.float64(38.5), np.float64(32.0)),
                        ),
                    ),
                ),
            ]
        ]
        assert output.width == 512
        assert output.height == 960


@pytest.mark.parametrize("confs", [[0.1], [0.1, 0.3]])
def test_object_detection_pipeline_w_postprocessing_params(confs):
    """Test enabling postprocessing parameters."""

    postprocessing_params = {"nms": [NMSConfig(conf=c) for c in confs]}

    # Mock the model to be used in the pipeline
    mock_model = MagicMock()
    mock_predict_return = torch.rand((1, 6, 6))
    mock_model.predict.return_value = mock_predict_return

    # Patch the _load_model method to return the mocked model
    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(model=model_cfg)
        batch_size = dataset_cfg.batch_size

        pipeline = ObjectDetectionPipeline(config, dataset_cfg, postprocessing_params)
        # Minus 1 for optical flow
        batch_size = (
            pipeline.metadata.numframes - 1
            if pipeline.metadata.numframes < batch_size
            else batch_size
        )
        pipeline.model = mock_model
        output = pipeline()

        assert mock_model.predict.call_count == batch_size

        # Get the postprocessing parameters
        steps = pipeline.postprocessing_steps

        assert len(output) == len(steps)
        assert len(steps) == len(confs)
        for step, conf_val in zip(steps, confs):
            assert callable(step)
            assert step.keywords["nms_config"].conf == conf_val


@pytest.mark.parametrize(
    "param",
    [
        {"nms": NMSConfig()},
        {"nms": [NMSConfig()]},
        {"nms": []},
    ],
)
def test_object_detection_pipeline_diff_postprocessing_structure(param):
    """Test enabling postprocessing parameters."""

    postprocessing_params = param

    # Mock the model to be used in the pipeline
    mock_model = MagicMock()
    mock_predict_return = torch.rand((1, 6, 6))
    mock_model.predict.return_value = mock_predict_return

    # Patch the _load_model method to return the mocked model
    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(model=model_cfg)
        batch_size = dataset_cfg.batch_size

        pipeline = ObjectDetectionPipeline(config, dataset_cfg, postprocessing_params)
        # Minus 1 for optical flow
        batch_size = (
            pipeline.metadata.numframes - 1
            if pipeline.metadata.numframes < batch_size
            else batch_size
        )
        pipeline.model = mock_model
        output = pipeline()

        assert mock_model.predict.call_count == batch_size
        # Get the postprocessing parameters
        steps = pipeline.postprocessing_steps

        if steps:
            assert len(output) == len(steps)


def test_object_detection_pipeline_postprocessing_invalid_params():
    """Test sending invalid postprocessing parameters"""

    # Mock the model to be used in the pipeline
    mock_model = MagicMock()
    mock_predict_return = torch.rand((1, 6, 6))
    mock_model.predict.return_value = mock_predict_return

    # Patch the _load_model method to return the mocked model
    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(model=model_cfg)
        processing_params = {"some_func_name": "some_func"}
        with pytest.raises(
            ValueError, match="Unknown postprocessing step: some_func_name"
        ):
            ObjectDetectionPipeline(config, dataset_cfg, processing_params)

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from conftest import ARIS_FILE
from fisheye.configs import (
    ObjectDetectionConfig,
    ObjectDetectionPipelineOutput,
    YOLODatasetConfig,
    YOLOv5ModelConfig,
)
from fisheye.configs.inference import NMSConfig
from fisheye.detect.yolov5 import YOLOv5ObjectDetectionModel
from fisheye.pipelines import ObjectDetectionPipeline


@pytest.fixture
def mock_yolov5_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = (
        torch.rand((1, 6)),
        None,
        None,
        None,
    )

    return mock_model


@pytest.fixture
def mock_pipeline(mock_yolov5_model):
    """Fixture for a mocked ObjectDetectionPipeline."""
    with patch("yolov5.load", return_value=mock_yolov5_model) as mock_load:
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(model=model_cfg)
        pipeline = ObjectDetectionPipeline(
            config=config, dataset_config=YOLODatasetConfig(filepath=ARIS_FILE)
        )

    pipeline.dataloader = MagicMock()
    pipeline.dataset = MagicMock()
    pipeline.run = MagicMock(
        return_value=ObjectDetectionPipelineOutput(
            pred_bboxes=[torch.rand((1, 6))],
            image_shape=[[1, 2, 3]],
            width=640,
            height=480,
        )
    )

    return pipeline


def test_initialization(mock_pipeline):
    """Test pipeline initialization."""
    assert isinstance(mock_pipeline, ObjectDetectionPipeline)
    assert mock_pipeline.model is not None
    assert mock_pipeline.device == "cpu"
    assert isinstance(mock_pipeline.dataloader, MagicMock)


def test_building_postprocessing_params(mock_pipeline):
    """Test building postprocessing parameters."""

    postprocessing_params = None
    sanitized_steps = mock_pipeline._build_postprocessing_params(postprocessing_params)
    assert len(sanitized_steps) == 0
    assert not callable(sanitized_steps)

    postprocessing_params = {"nms": {"nms_config": NMSConfig()}}
    sanitized_steps = mock_pipeline._build_postprocessing_params(postprocessing_params)
    assert len(sanitized_steps) == 1
    assert callable(sanitized_steps[0])

    postprocessing_params = {"nms": [NMSConfig(conf=0.1), NMSConfig(conf=0.3)]}
    sanitized_steps = mock_pipeline._build_postprocessing_params(postprocessing_params)
    assert len(sanitized_steps) == 2
    print(sanitized_steps)
    assert callable(sanitized_steps[0])
    assert sanitized_steps[0].keywords["nms_config"].conf == 0.1
    assert callable(sanitized_steps[1])
    assert sanitized_steps[1].keywords["nms_config"].conf == 0.3


def test_object_detection_pipeline_w_postprocessing_params(mock_pipeline):
    """Test enabling postprocessing parameters."""

    postprocessing_params = {"nms": [NMSConfig(conf=0.1), NMSConfig(conf=0.3)]}

    # Mock the model to be used in the pipeline
    mock_model = MagicMock()
    mock_predict_return = torch.rand((3, 6, 6))
    mock_model.predict.return_value = mock_predict_return

    # Patch the _load_model method to return the mocked model
    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(model=model_cfg)

        pipeline = ObjectDetectionPipeline(config, dataset_cfg, postprocessing_params)
        pipeline.model = mock_model
        output = pipeline()

        mock_model.predict.assert_called_once()
        assert len(output) == 2


def test_preprocess(mock_pipeline):
    """Test image preprocessing."""
    mock_image = torch.rand((1, 3, 640, 640))  # Mock image tensor
    preprocessed_image = mock_pipeline.preprocess(mock_image)

    assert preprocessed_image.device == torch.device("cpu")
    assert preprocessed_image.shape == (1, 3, 640, 640)
    assert (preprocessed_image <= 1.0).all()  # Ensure the image is normalized


def test_run(mock_pipeline):
    """Test running the full pipeline (inference + postprocessing)."""
    mock_pipeline.run = MagicMock(
        return_value=ObjectDetectionPipelineOutput(
            pred_bboxes=[torch.rand((1, 6, 6))],
            image_shape=[[(640, 640), (640, 640)]],
            width=640,
            height=640,
        )
    )

    output = mock_pipeline.run()

    assert isinstance(output, ObjectDetectionPipelineOutput)
    assert isinstance(output.pred_bboxes, list)
    assert output.pred_bboxes[0].shape == (1, 6, 6)
    assert isinstance(output.image_shape, list)
    assert output.width == 640
    assert output.height == 640


def test_object_detection_pipeline():
    """Test ObjectDetectionPipeline with a mocked `self.model` (patched _load_model)."""

    # Mock the model to be used in the pipeline
    mock_model = MagicMock()
    mock_predict_return = torch.rand((3, 6, 6))
    mock_model.predict.return_value = mock_predict_return

    # Patch the _load_model method to return the mocked model
    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(model=model_cfg)

        pipeline = ObjectDetectionPipeline(config, dataset_cfg)
        pipeline.model = mock_model
        output = pipeline()

        mock_model.predict.assert_called_once()
        assert len(output.pred_bboxes) == 1
        assert isinstance(
            output.pred_bboxes[0][0], torch.Tensor
        )  # Check predictions are tensors
        assert output.pred_bboxes[0].shape == (
            3,
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


def test_object_detection_pipeline_postprocessing_invalid_params(mock_pipeline):
    """Test sending invalid postprocessing parameters"""

    # Mock the model to be used in the pipeline
    mock_model = MagicMock()
    mock_predict_return = torch.rand((3, 6, 6))
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

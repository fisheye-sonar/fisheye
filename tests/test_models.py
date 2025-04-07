from unittest.mock import MagicMock, patch

import pytest
import torch

from fisheye.configs import YOLODatasetConfig, ObjectDetectionConfig
from fisheye.detect.yolov5 import YOLOv5ObjectDetectionModel, YOLOv5ModelConfig
from fisheye.pipelines import ObjectDetectionPipeline
from conftest import ARIS_FILE


@pytest.fixture
def mock_yolov5_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = (
        torch.rand((1, 6)),
        None,
        None,
        None,
    )  # Mocked inference output

    return mock_model


def test_loading_yolov5(mock_yolov5_model):
    """Mock model and confirm configs from YOLOv5ModelConfig are correctly loaded."""
    with patch("yolov5.load", return_value=mock_yolov5_model) as mock_load:
        # Changing a couple default configs to make sure
        config = YOLOv5ModelConfig(weights="dummy/path")
        detector = YOLOv5ObjectDetectionModel(config)

        # Assertions to verify correct behavior - ensure yolov5.load was called
        mock_load.assert_called_once_with("dummy/path", "cpu")
        assert detector.model.agnostic == config.agnostic
        assert detector.model.multi_label == config.multi_label
        assert detector.model.classes == config.classes
        assert detector.model.max_det == config.max_det
        assert detector.model.amp == config.amp


def test_yolov5_predict():
    """Test making predictions using the YOLOv5 model directly with mocked output."""

    mock_model = MagicMock()

    # mock the `__call__` method of the mock model to return a torch tensor
    mock_model.return_value = torch.rand((1, 6))
    dummy_input = torch.rand((1, 3, 640, 640))
    config = YOLOv5ModelConfig(weights="dummy/path")

    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        # Instantiate the detector using the patched _load_model method
        detector = YOLOv5ObjectDetectionModel(config)
        detector.model = mock_model

        predictions = detector(dummy_input)

        # Assertions
        mock_model.assert_called_once_with(
            dummy_input
        )  # Check if the model was called correctly
        assert isinstance(
            predictions, torch.Tensor
        ), f"Expected torch.Tensor but got {type(predictions)}"
        assert predictions.shape == (
            1,
            6,
        ), f"Expected shape (1, 6) but got {predictions.shape}"


def test_object_detection_pipeline(mock_yolov5_model):
    """Test ObjectDetectionPipeline with a mocked YOLOv5 model and `_forward` method."""
    # Mock the `_forward` method to return dummy data
    mock_forward_return = (
        [[torch.rand((1, 6, 6))]],  # bbox pred
        [[((640, 640), (640, 640))]],  # image shapes
        640,  # width
        640,  # height
    )

    with patch(
        "yolov5.load", return_value=mock_yolov5_model
    ) as mock_load, patch.object(
        ObjectDetectionPipeline, "_forward", return_value=mock_forward_return
    ):
        dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(model=model_cfg)

        output = ObjectDetectionPipeline(config, dataset_cfg).run()

        # Assertions to verify behavior
        mock_load.assert_called_once_with("dummy/path", "cpu")
        assert len(output.pred_bboxes) == 1  # Only one batch of predictions
        assert isinstance(
            output.pred_bboxes[0][0], torch.Tensor
        )  # Check predictions are tensors
        assert output.pred_bboxes[0][0].shape == (1, 6, 6)
        assert output.image_shape == [[((640, 640), (640, 640))]]
        assert output.width == 640
        assert output.height == 640

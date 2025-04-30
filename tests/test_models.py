from unittest.mock import MagicMock, patch

import pytest
import torch

from fisheye.detect.yolov5 import YOLOv5ObjectDetectionModel, YOLOv5ModelConfig


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
        mock_load.assert_called_once_with("dummy/path", config.device)
        assert detector.model.agnostic == config.agnostic
        assert detector.model.multi_label == config.multi_label
        assert detector.model.classes == config.classes
        assert detector.model.max_det == config.max_det
        assert detector.model.amp == config.amp


def test_yolov5_predict():
    """Test making predictions using the YOLOv5 model directly with mocked output."""

    mock_model = MagicMock()

    # mock the `__call__` method of the mock model to return a torch tensor
    mock_model.return_value = torch.rand((3, 6, 6))
    dummy_input = torch.rand((3, 3, 640, 640))
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
            3,
            6,
            6,
        ), f"Expected shape (3, 6, 6) but got {predictions.shape}"

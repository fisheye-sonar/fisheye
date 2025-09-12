from unittest.mock import MagicMock, patch

import pytest
import torch

from fisheye.detect.yolov5 import YOLOv5ObjectDetectionModel, YOLOv5ModelConfig
from fisheye.detect.yolov11 import YOLOv11ObjectDetectionModel, YOLOv11ModelConfig


@pytest.fixture
def mock_yolo_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = (
        torch.rand((1, 6)),
        None,
        None,
        None,
    )  # Mocked inference output

    return mock_model


def test_loading_yolov5(mock_yolo_model):
    """Mock model and confirm configs from YOLOv5ModelConfig are correctly loaded."""
    with patch("yolov5.load", return_value=mock_yolo_model) as mock_load:
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


def test_loading_yolov11(mock_yolo_model):
    """Mock YOLO class and confirm YOLOv11ObjectDetectionModel loads correctly."""
    with patch("fisheye.detect.yolov11.YOLO", autospec=True) as mock_yolo_cls:
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.model = mock_yolo_model
        mock_yolo_cls.return_value = mock_yolo_instance

        # Build config + detector
        config = YOLOv11ModelConfig(weights="dummy/path")
        detector = YOLOv11ObjectDetectionModel(config)

        # Ensure YOLO constructor was called correctly
        mock_yolo_cls.assert_called_once_with("dummy/path")

        # Verify detector.model is our mocked model
        assert detector.model is mock_yolo_model


def test_yolov11_predict():
    """Test making predictions using the YOLOv11 model with mocked output."""
    mock_model = MagicMock()

    # Mock the return value of model(images)
    # For a single class, YOLOv11 raw output is (x1, y1, x2, y2, obj_conf) = 5 fields
    # [B, 5, N]
    fake_prediction = torch.rand((3, 5, 8))
    mock_model.return_value = (fake_prediction, None)

    dummy_input = torch.rand((3, 3, 640, 640))
    config = YOLOv11ModelConfig(weights="dummy/path")

    with patch.object(
        YOLOv11ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        detector = YOLOv11ObjectDetectionModel(config)
        detector.model = mock_model

        predictions = detector.predict(dummy_input)

        mock_model.assert_called_once_with(
            dummy_input
        )  # Model should be called with the input
        assert isinstance(
            predictions, torch.Tensor
        ), f"Expected torch.Tensor but got {type(predictions)}"

        # Expect shape [B, N, 6] - YOLOv11 returns 5 fields but we add on class_idx as an extra field for NMS + filtering
        assert (
            predictions.shape[0] == 3
        ), f"Expected batch size 3 but got {predictions.shape[0]}"
        assert (
            predictions.shape[2] == 6
        ), f"Expected last dim=6 but got {predictions.shape[2]}"

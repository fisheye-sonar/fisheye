from unittest.mock import MagicMock, patch

import pytest
import torch

from fisheye.dataclasses import YOLODatasetConfig
from fisheye.models.yolov5 import YOLOv5ObjectDetectionModel, YOLOv5ModelConfig
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
        config = YOLOv5ModelConfig(model="dummy/path", conf=0.5, iou=0.45)
        detector = YOLOv5ObjectDetectionModel(config)

        # Assertions to verify correct behavior - ensure yolov5.load was called
        mock_load.assert_called_once_with("dummy/path", "cpu")
        assert detector.model.conf == 0.5
        assert detector.model.iou == 0.45
        assert detector.model.agnostic == config.agnostic
        assert detector.model.multi_label == config.multi_label
        assert detector.model.classes == config.classes
        assert detector.model.max_det == config.max_det
        assert detector.model.amp == config.amp


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
        model_cfg = YOLOv5ModelConfig(model="dummy/path", conf=0.5, iou=0.45)
        output = ObjectDetectionPipeline(model_cfg, dataset_cfg).run()

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

import pytest
import torch
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from fisheye.lengths.estimator import UNetLengthEstimator
from fisheye.configs.models import UNetLengthModelConfig


@pytest.fixture
def mock_metadata():
    return SimpleNamespace(
        xdim=100,
        ydim=100,
        pixel_meter_size=0.01,
        unwarped_shape=(100, 100),
        sampleperiod=30,
        soundspeed=1500,
        window_start=0,
        window_length=100,
        samplesperbeam=512,
        beams=96,
        windowstart=0,
        windowlength=100,
    )


@pytest.fixture
def mock_config():
    config = UNetLengthModelConfig()
    config.device = "cpu"
    config.weights = None  # No weights for testing
    return config


@pytest.fixture
def estimator(mock_metadata, mock_config):
    with patch(
        "fisheye.lengths.estimator.get_cone_edges",
        return_value=(None, None, None, None),
    ):
        return UNetLengthEstimator(mock_metadata, mock_config)


def test_initialization(estimator):
    """Test that the estimator initializes correctly."""
    assert estimator.model is not None
    assert estimator.config.device == "cpu"


def test_get_pred_from_img(estimator):
    """Test prediction from a single image."""
    # Mock model output
    estimator.model = MagicMock(return_value=torch.randn(1, 2, 50, 50))

    img = torch.randn(1, 3, 100, 100)
    crop_ltrbs = [[10, 10, 10, 10]]  # [l, t, r, b] padding from edges

    outputs = estimator.get_pred_from_img(img, crop_ltrbs)

    assert len(outputs) == 1
    assert "pred_kpts_global_px" in outputs[0]

    # Check that model was called
    estimator.model.assert_called()


def test_get_pred_from_batch(estimator):
    """Test prediction from a batch of frames."""
    estimator.model = MagicMock(return_value=torch.randn(1, 2, 50, 50))

    frames_batch = [torch.randn(3, 100, 100) for _ in range(2)]
    crop_info = [
        {"frame_num": 0, "crop_ltrbs": [[10, 10, 10, 10]]},
        {"frame_num": 1, "crop_ltrbs": []},  # No crops for second frame
    ]

    outputs = estimator.get_pred_from_batch(crop_info, frames_batch)

    assert 0 in outputs
    assert 1 not in outputs
    assert len(outputs[0]) == 1


def test_get_crop_info(estimator):
    """Test converting bboxes to crop info."""
    pred_bboxes_batch = {
        (0, 0): [[0.1, 0.1, 0.2, 0.2]],  # [x1, y1, x2, y2] normalized
        (0, 1): None,
    }

    crop_info = estimator.get_crop_info(pred_bboxes_batch)

    assert len(crop_info) == 1
    assert crop_info[0]["frame_num"] == 0
    assert len(crop_info[0]["crop_ltrbs"]) == 1


def test_run(estimator):
    """Test the main run method."""
    # Mock get_length_estimates
    estimator.get_length_estimates = MagicMock(return_value={"result": "ok"})

    frames_batch = [torch.randn(3, 100, 100)]
    pred_bboxes = {(0, 0): [[0.1, 0.1, 0.2, 0.2]]}

    result = estimator.run(frames_batch, pred_bboxes)

    assert result == {"result": "ok"}
    estimator.get_length_estimates.assert_called_once_with(frames_batch, pred_bboxes)

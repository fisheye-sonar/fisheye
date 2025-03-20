import numpy as np
import pytest

from conftest import ARIS_FILE
from fisheye.configs import YOLODatasetConfig
from fisheye.configs.inference import TrackerConfig
from fisheye.track.tracker import run_tracker


# Sample predictions
low_preds = {
    (0, 0): np.array([[0.1, 0.2, 0.3, 0.4, 0.95]]),
    (0, 1): np.array([[0.2, 0.3, 0.4, 0.5, 0.85]]),
}

high_preds = {
    (0, 0): np.array([[0.1, 0.2, 0.3, 0.4, 0.99]]),
    (0, 1): np.array([[0.2, 0.3, 0.4, 0.5, 0.88]]),
}


def test_run_tracker_bytetrack():
    """Test basic run_tracker functionality."""

    tracking_config = TrackerConfig()
    config = YOLODatasetConfig(filepath=ARIS_FILE)

    output = run_tracker(
        low_preds,
        high_preds,
        config.image_meter_width,
        config.image_meter_height,
        tracking_config,
    )

    assert len(output["frames"]) == len(low_preds) and len(output["frames"]) == len(
        high_preds
    ), (
        f"Expected {len(high_preds)} frames to be processed, "
        f"but got {len(output['frames'])}"
    )


def test_run_tracker_with_empty_predictions():
    """Test run_tracker with empty predictions and check how many frames were processed."""
    # Define empty predictions and None case
    empty_preds = {(0, 0): np.empty((0, 5)), (0, 1): None}

    tracking_config = TrackerConfig()
    config = YOLODatasetConfig(filepath=ARIS_FILE)

    # Run the tracker
    output = run_tracker(
        empty_preds,
        empty_preds,
        config.image_meter_width,
        config.image_meter_height,
        tracking_config,
    )

    # Check how many frames were processed
    assert len(output["frames"]) == len(empty_preds), (
        f"Expected {len(empty_preds)} frames to be processed, "
        f"but got {len(output['frames'])}"
    )


def test_invalid_tracker_types():
    """Test run_tracker invalid tracker"""

    tracking_config = TrackerConfig(type="other_tracker")
    config = YOLODatasetConfig(filepath=ARIS_FILE)

    with pytest.raises(
        ValueError, match="Tracking method `other_tracker` is not supported."
    ):
        run_tracker(
            low_preds,
            high_preds,
            config.image_meter_width,
            config.image_meter_height,
            tracking_config,
        )

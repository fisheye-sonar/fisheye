import numpy as np
import pytest

from conftest import ARIS_FILE
from fisheye.configs import YOLODatasetConfig
from fisheye.configs.inference import TrackerConfig
from fisheye.track.tracker import run_tracker


# Sample predictions
low_preds = {
    (0, 0): np.array([[0.82594, 0.30592, 0.85993, 0.31484, 0.19332]]),
    (0, 1): np.array([[0.82542, 0.30521, 0.85904, 0.3149, 0.42544]]),
    (0, 2): np.array([[0.83115, 0.30476, 0.85935, 0.31404, 0.23077]]),
    (0, 3): np.array([[0.83448, 0.30276, 0.8612, 0.31195, 0.28912]]),
    (0, 4): np.array([[0.52299, 0.38605, 0.56787, 0.39254, 0.11451]]),
    (0, 5): None,
    (0, 6): None,
    (1, 0): None,
    (1, 1): None,
    (1, 2): None,
    (1, 3): None,
    (1, 4): None,
    (1, 5): None,
    (1, 6): np.array([[0.77465, 0.34709, 0.82632, 0.35651, 0.39355]]),
}

high_preds = {
    (0, 0): None,
    (0, 1): np.array([[0.82542, 0.30521, 0.85904, 0.3149, 0.42544]]),
    (0, 2): None,
    (0, 3): None,
    (0, 4): None,
    (0, 5): None,
    (0, 6): None,
    (1, 0): None,
    (1, 1): None,
    (1, 2): None,
    (1, 3): None,
    (1, 4): None,
    (1, 5): None,
    (1, 6): np.array([[0.77465, 0.34709, 0.82632, 0.35651, 0.39355]]),
}

empty_preds = {(0, 0): np.empty((0, 5)), (0, 1): None}


@pytest.mark.parametrize(
    "low_preds, high_preds, expected_frames",
    [
        (low_preds, high_preds, len(low_preds)),
        (empty_preds, empty_preds, len(empty_preds)),
    ],
)
def test_run_tracker_bytetrack(low_preds, high_preds, expected_frames):
    """Test basic run_tracker functionality."""

    # Lowering min_hits to ensure test cases trigger relevant code blocks
    tracking_config = TrackerConfig(min_hits=1)

    output = run_tracker(
        low_preds,
        high_preds,
        11.368606870117187,
        23.060770468749997,
        tracking_config,
    )

    assert len(output.frames) == expected_frames, (
        f"Expected {expected_frames} frames to be processed, "
        f"but got {len(output.frames)}"
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

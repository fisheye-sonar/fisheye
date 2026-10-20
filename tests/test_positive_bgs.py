"""Opt-in positive-only background subtraction preserves default model inputs."""

import numpy as np
import pytest

from fisheye.configs import BaseDatasetConfig
from fisheye.dataloaders.base import BaseDataset


class PreprocessingDataset(BaseDataset):
    def _init_bg_frame(self):
        pass


@pytest.mark.parametrize(
    "positive_only,expected,difference",
    [
        (None, [[0, 127], [127, 255]], [[127, 127], [127, 127]]),
        (False, [[0, 127], [127, 255]], [[127, 127], [127, 127]]),
        (True, [[0, 0], [0, 255]], [[0, 255], [255, 255]]),
    ],
)
def test_positive_only_background_subtraction(positive_only, expected, difference):
    frames = np.array([[[0, 10]], [[10, 20]], [[20, 30]]], dtype=np.uint8)
    config = BaseDatasetConfig(use_blur=False)
    if positive_only is not None:
        config.only_pos_bgs = positive_only
    ds = PreprocessingDataset(config)
    result = ds._stack_preprocessed_channels(
        frames, np.array([[10, 10]], dtype=np.float32), 10.0
    )
    np.testing.assert_array_equal(result[:, 0, :, 0], frames[:-1, 0])
    np.testing.assert_array_equal(result[:, 0, :, 1], expected)
    np.testing.assert_array_equal(result[:, 0, :, 2], difference)
    assert result.dtype == np.uint8


def test_positive_only_clips_before_temporal_difference_and_saturates():
    frames = np.array([0, 10, 20, 25, 30, 40], dtype=np.uint8).reshape(-1, 1, 1)
    ds = PreprocessingDataset(BaseDatasetConfig(use_blur=False, only_pos_bgs=True))
    result = ds._stack_preprocessed_channels(frames, 20.0, 10.0)

    np.testing.assert_array_equal(result[:, 0, 0, 0], frames[:-1, 0, 0])
    np.testing.assert_array_equal(result[:, 0, 0, 1], [0, 0, 0, 127, 255])
    np.testing.assert_array_equal(result[:, 0, 0, 2], [0, 0, 127, 127, 255])

"""Numerical and output-contract regressions using small deterministic recordings."""

from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

from fisheye.configs import BaseDatasetConfig, YOLODatasetConfig
from fisheye.dataloaders.base import BaseDataset
from fisheye.dataloaders.compute_bg_subtraction import compute_bg_subtraction
from fisheye.dataloaders.didson.pyDIDSON import DIDSON
from fisheye.dataloaders.echogram import compute_echogram
from fisheye.dataloaders.yolo import YOLOARISBatchedDataset


FRAMES = np.arange(9 * 4 * 3, dtype=np.uint8).reshape(9, 4, 3)


class SmallDataset(BaseDataset):
    def load_frames(self, idx, final_idx, return_unwarped=False, return_warped=True):
        return (
            FRAMES[idx:final_idx, :, :2] if return_warped else None,
            FRAMES[idx:final_idx] if return_unwarped else None,
        )


def config(**kwargs):
    return BaseDatasetConfig(**{"end_frame": 9, "num_frames_bg_subtract": 3, **kwargs})


def test_channel_values_and_order():
    frames = np.array([[[0, 64, 32]], [[0, 128, 16]]], dtype=np.uint8)
    result = compute_echogram(
        frames,
        np.zeros((1, 3)),
        128,
        echogram_channels=[
            "raw",
            "center_line",
            "angle",
            "bgs",
            "bgs_angle",
            "0",
            None,
        ],
    )
    np.testing.assert_allclose(result[:, 0, 0], [64 / 255, 128 / 255])
    np.testing.assert_allclose(result[:, 0, 1], [64 / 255, 128 / 255])
    np.testing.assert_allclose(result[:, 0, 2], [-1 / 6, -1 / 6])
    np.testing.assert_allclose(result[:, 0, 3], [0.5, 1])
    np.testing.assert_allclose(result[:, 0, 4], [-1 / 6, -1 / 6])
    assert np.all(result[..., 5] == 0)
    assert result.dtype == np.float32


@pytest.mark.parametrize("channels", [[], [None], [None, "raw"], ["invalid"]])
def test_invalid_channels(channels):
    with pytest.raises(ValueError):
        BaseDatasetConfig(echogram_channels=channels)
    with pytest.raises(ValueError):
        compute_echogram(FRAMES, echogram_channels=channels)


@pytest.mark.parametrize(
    "use_blur,use_multithreading", product([False, True], repeat=2)
)
def test_constant_background_is_finite(use_blur, use_multithreading):
    frames = np.full((3, 4, 3), 37, dtype=np.uint8)
    mean, scale = compute_bg_subtraction(frames, use_blur, use_multithreading)
    echo = compute_echogram(frames, mean, scale)
    assert np.isfinite(echo).all()
    np.testing.assert_array_equal(echo[..., 0], 0)
    ds = SmallDataset(config(use_blur=use_blur, use_multithreading=use_multithreading))
    image = ds._stack_preprocessed_channels(frames, mean, scale)
    np.testing.assert_array_equal(image[..., 1], 127)
    np.testing.assert_array_equal(image[..., 2], 0)


@pytest.mark.parametrize(
    "warped,unwarped,echo", [p for p in product([False, True], repeat=3) if any(p)]
)
@pytest.mark.parametrize("bgs,blur,threading", product([False, True], repeat=3))
def test_independent_output_matrix(warped, unwarped, echo, bgs, blur, threading):
    ds = SmallDataset(
        config(
            return_frames=warped,
            return_unwarped=unwarped,
            return_echogram=echo,
            do_bg_subtract=bgs,
            use_blur=blur,
            use_multithreading=threading,
            batch_size=3,
        )
    )
    images, labels, raw, echogram, original = ds[0]
    assert (images is not None) == warped
    assert (raw is not None) == unwarped
    assert (echogram is not None) == echo
    assert labels is None
    for value in (images, raw, echogram):
        if value is not None:
            assert value.shape[0] == 3
            assert np.isfinite(value).all()
    if raw is not None:
        assert raw.shape == ((3, 4, 3, 3) if bgs else (3, 4, 3))
    if not warped:
        assert original is False


@pytest.mark.parametrize(
    "outputs",
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ],
)
@pytest.mark.parametrize("start_frame", [0, 2])
def test_cache_tracks_frame_indices_and_partial_batches(outputs, start_frame):
    options = dict(
        return_frames=outputs[0],
        return_unwarped=outputs[1],
        return_echogram=outputs[2],
        batch_size=3,
        start_frame=start_frame,
        use_blur=False,
    )
    cached = SmallDataset(config(cache_bg_frames=True, **options))
    reference = SmallDataset(config(**options))
    for idx in [3, 0, 3, len(cached) - 1, 1, 0]:
        actual = cached[idx]
        expected = reference[idx]
        for slot in [0, 2, 3]:
            if expected[slot] is None:
                assert actual[slot] is None
            else:
                np.testing.assert_array_equal(actual[slot], expected[slot])


def test_legacy_unwarped_primary_and_raw_auxiliary():
    ds = SmallDataset(config(return_unwarped=True, use_blur=False))
    primary, _, auxiliary, _, _ = ds[0]
    assert primary.shape == (8, 4, 3, 3)
    np.testing.assert_array_equal(auxiliary, FRAMES[:-1])


def test_legacy_raw_echogram_scale_and_explicit_precedence():
    ds = SmallDataset(
        config(
            return_frames=False,
            return_echogram=True,
            return_echogram_with_bg_subtracted=False,
        )
    )
    np.testing.assert_allclose(ds[0][3][..., 0], FRAMES[:-1].max(axis=2))
    explicit = SmallDataset(
        config(
            return_frames=False,
            return_echogram=True,
            return_echogram_with_bg_subtracted=False,
            echogram_channels=["raw"],
        )
    )
    np.testing.assert_allclose(explicit[0][3][..., 0], FRAMES[:-1].max(axis=2) / 255)


def test_echogram_only_skips_warping_and_raw_skips_background():
    with patch.object(SmallDataset, "load_frames", autospec=True, wraps=None) as load:
        load.return_value = (None, FRAMES)
        ds = SmallDataset(
            config(return_frames=False, return_echogram=True, echogram_channels=["raw"])
        )
        assert not load.called
        ds[0]
        assert load.call_count == 1
        assert load.call_args.kwargs == {
            "return_unwarped": True,
            "return_warped": False,
        }


def test_yolo_rejects_echogram_only_before_opening_file():
    with pytest.raises(ValueError, match="YOLO requires return_frames=True"):
        YOLOARISBatchedDataset(
            YOLODatasetConfig(
                filepath="does-not-exist", return_frames=False, return_echogram=True
            )
        )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_unwarped_background_count_is_batch_independent(batch_size):
    ds = SmallDataset(
        config(
            return_frames=False,
            return_echogram=True,
            batch_size=batch_size,
            use_blur=False,
        )
    )
    mean, scale = compute_bg_subtraction(FRAMES[:3], use_blur=False)
    np.testing.assert_allclose(ds.unwarped_mean_blurred_frame, mean)
    assert ds.unwarped_mean_normalization_value == scale


def test_empty_dataset_and_bounds():
    ds = SmallDataset(BaseDatasetConfig(start_frame=2, end_frame=2))
    assert len(ds) == 0
    with pytest.raises(IndexError):
        ds[0]
    ds = SmallDataset(config())
    for idx in [-1, len(ds)]:
        with pytest.raises(IndexError):
            ds[idx]


@pytest.mark.parametrize("batch_size,chunk_size", [(1, 1), (2, 3), (4, 256)])
@pytest.mark.parametrize("start_frame,end_frame", [(0, 9), (2, 8), (3, 4), (4, 4)])
def test_direct_chunked_loader_matches_dataset(
    batch_size, chunk_size, start_frame, end_frame
):
    didson = DIDSON.__new__(DIDSON)
    didson.info = {"numframes": 9, "unwarped_shape": (4, 3)}
    calls = []

    def load(file=None, start_frame=0, end_frame=-1, **kwargs):
        assert kwargs == {"return_unwarped": True, "return_warped": False}
        calls.append((start_frame, end_frame))
        return None, FRAMES[start_frame:end_frame]

    didson.load_frames = load
    actual = didson.load_echogram(
        start_frame=start_frame,
        end_frame=end_frame,
        num_frames_bg_subtract=3,
        use_blur=False,
        chunk_size=chunk_size,
    )
    ds = SmallDataset(
        config(
            start_frame=start_frame,
            end_frame=end_frame,
            return_frames=False,
            return_echogram=True,
            batch_size=batch_size,
            use_blur=False,
        )
    )
    if len(ds):
        expected = np.concatenate([ds[idx][3] for idx in range(0, len(ds), batch_size)])
        np.testing.assert_allclose(actual, expected)
        assert all(end - start <= chunk_size for start, end in calls[1:])
    else:
        assert actual.shape == (0, 4, 2)
        assert not calls


@pytest.mark.parametrize("start,end", [(-2, 4), (5, 4), (0, 10)])
def test_direct_loader_rejects_invalid_range(start, end):
    didson = DIDSON.__new__(DIDSON)
    didson.info = {"numframes": 9, "unwarped_shape": (4, 3)}
    with pytest.raises(ValueError, match="start_frame"):
        didson.load_echogram(start_frame=start, end_frame=end)

"""Native detector-channel capture and source identity checks."""

from copy import deepcopy
from multiprocessing.connection import Listener
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fisheye.common.collate import yolo_collate_fn
from fisheye.configs import YOLODatasetConfig
from fisheye.dataloaders import YOLOARISBatchedDataset, create_dataloader
from fisheye.dataloaders.base import BaseDataset
from fisheye.dataloaders.samplers import OnePerBatchSampler


class SyntheticCaptureDataset(YOLOARISBatchedDataset):
    """Use real preprocessing/collation with a deterministic decoded frame source."""

    def __init__(self, frames, config):
        self.raw_frames = frames
        self.reads = []
        self.metadata = SimpleNamespace(ydim=frames.shape[1], xdim=frames.shape[2])
        BaseDataset.__init__(self, config)
        self.stride, self.pad, self.img_size = (
            config.stride,
            config.pad,
            config.img_size,
        )
        self.original_shape = frames.shape[1:]
        self.shape = self._compute_resized_shape()

    def load_frames(self, start_frame, end_frame, return_unwarped=False):
        self.reads.append((start_frame, end_frame))
        frames = self.raw_frames[start_frame:end_frame].copy()
        return frames, frames.copy() if return_unwarped else None

    @classmethod
    def load_image(cls, image, img_size=32, return_original_image=False):
        return super().load_image(image, img_size, return_original_image)


class SyntheticBaseCaptureDataset(SyntheticCaptureDataset):
    _postprocess = BaseDataset._postprocess


@pytest.fixture
def source_frames():
    frames = np.full((9, 10, 8), 20, dtype=np.uint8)
    frames[:, 3, 4] = np.arange(9, dtype=np.uint8) * 20
    frames[:, 6, 2] = np.arange(9, dtype=np.uint8)[::-1] * 10
    return frames


def synthetic_config(**kwargs):
    values = dict(
        start_frame=2,
        end_frame=9,
        batch_size=4,
        img_size=32,
        stride=8,
        use_multithreading=False,
        capture_endpoint_frames=True,
        use_blur=False,
        only_pos_bgs=False,
    )
    values.update(kwargs)
    return YOLODatasetConfig(**values)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("cache", [False, True])
def test_frame_ids_partial_batch_cache_and_workers(source_frames, workers, cache):
    if workers:
        # Torch shares tensor file descriptors through a local multiprocessing
        # listener, which restricted execution sandboxes may forbid.
        try:
            listener = Listener()
        except PermissionError:
            pytest.skip("Sandbox prohibits multiprocessing tensor-sharing sockets")
        else:
            listener.close()
    ds = SyntheticCaptureDataset(source_frames, synthetic_config(cache_bg_frames=cache))
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=workers,
        sampler=OnePerBatchSampler(ds, ds.batch_size),
        collate_fn=yolo_collate_fn,
        multiprocessing_context="spawn" if workers else None,
        persistent_workers=bool(workers),
        timeout=30 if workers else 0,
    )
    batches = list(loader)
    assert [batch[4]["frame_ids"] for batch in batches] == [(2, 3, 4, 5), (6, 7)]
    assert [len(batch[4]["images"]) for batch in batches] == [4, 2]
    assert sum(batch[0].shape[0] for batch in batches) == len(ds) == 6
    for first, repeated in zip(batches, loader):
        assert first[4]["frame_ids"] == repeated[4]["frame_ids"]
        assert torch.equal(first[0], repeated[0])
        np.testing.assert_array_equal(
            np.stack(first[4]["images"]), np.stack(repeated[4]["images"])
        )


@pytest.mark.parametrize("capture", [False, True])
def test_cache_supports_noncontiguous_requests(source_frames, capture):
    config = synthetic_config(cache_bg_frames=True, capture_endpoint_frames=capture)
    if capture:
        with pytest.warns(UserWarning, match="cache_bg_frames is disabled"):
            ds = SyntheticCaptureDataset(source_frames, config)
    else:
        ds = SyntheticCaptureDataset(source_frames, config)
    tail = yolo_collate_fn(ds[4])
    head = yolo_collate_fn(ds[0])
    read_count = len(ds.reads)
    again = yolo_collate_fn(ds[4])
    assert len(ds.reads) == read_count + int(capture)
    if capture:
        assert ds._frame_cache == {}
        assert tail[4]["frame_ids"] == again[4]["frame_ids"] == (6, 7)
        assert head[4]["frame_ids"] == (2, 3, 4, 5)
    assert torch.equal(tail[0], again[0])


def test_endpoint_store_spill_does_not_leave_an_unbounded_dataset_copy(tmp_path):
    from fisheye.endpoints.frame_store import BoundedFrameStore

    frames = np.random.default_rng(4).integers(0, 255, (31, 10, 8), dtype=np.uint8)
    with pytest.warns(UserWarning, match="cache_bg_frames is disabled"):
        ds = SyntheticBaseCaptureDataset(
            frames,
            synthetic_config(
                start_frame=0,
                end_frame=31,
                batch_size=2,
                cache_bg_frames=True,
            ),
        )
    with BoundedFrameStore("clip", ram_limit_bytes=0, directory=tmp_path) as store:
        for start in range(0, len(ds), 2):
            capture = ds[start][-1]
            for frame_id, image in zip(capture["frame_ids"], capture["images"]):
                store.put(frame_id, image)
            assert ds._frame_cache == {}
            assert store.ram_bytes == 0
        assert len(store.frame_ids) == 30
        assert store.disk_bytes > 0


def test_capture_uses_detector_background_once(source_frames, monkeypatch):
    def fail_blur(*args, **kwargs):
        raise AssertionError("Signed endpoint capture must not blur")

    monkeypatch.setattr(cv2, "GaussianBlur", fail_blur)
    with patch.object(
        BaseDataset,
        "_compute_bg_subtraction",
        autospec=True,
        side_effect=BaseDataset._compute_bg_subtraction,
    ) as compute:
        ds = SyntheticBaseCaptureDataset(source_frames, synthetic_config())
        ds[0]
        ds[4]
    assert compute.call_count == 1


def test_decoder_reset_is_rejected_instead_of_mislabeled(source_frames):
    ds = SyntheticCaptureDataset(source_frames, synthetic_config())
    ds.didson = SimpleNamespace(last_loaded_frame_ids=(0, 1, 2, 3, 4))
    with pytest.raises(ValueError, match="alignment changed"):
        ds[0]


def test_real_aris_native_capture_and_detector_parity():
    fixture = Path(__file__).parent / "data" / "sample.aris"
    config = YOLODatasetConfig(
        filepath=str(fixture),
        start_frame=1,
        batch_size=1,
        num_frames_bg_subtract=3,
        workers=0,
        use_multithreading=False,
        capture_endpoint_frames=True,
        use_blur=False,
        only_pos_bgs=False,
    )
    loader, ds = create_dataloader(config)
    captured = list(loader)
    raw, _ = ds.didson.load_frames(start_frame=1, end_frame=4)
    assert [tuple(batch[4]["frame_ids"]) for batch in captured] == [(1,), (2,)]
    assert captured[0][4]["images"][0].shape == (ds.metadata.ydim, ds.metadata.xdim)
    mean = np.mean(raw, axis=0)
    norm = np.max(raw.max(axis=0) - mean)
    expected = raw[:-1].astype(np.float32)
    expected -= mean
    expected /= norm
    expected += 1
    expected *= 0.5
    expected = (np.clip(expected, 0, 1) * 255).astype(np.uint8)
    np.testing.assert_array_equal(
        np.stack([batch[4]["images"][0] for batch in captured]), expected
    )
    assert captured[0][4]["preprocessing"]["background_frame_ids"] == [1, 2, 3]
    disabled = deepcopy(config)
    disabled.capture_endpoint_frames = False
    baseline, _ = create_dataloader(disabled)
    for capture, detector_only in zip(captured, baseline):
        assert len(detector_only) == 4
        assert torch.equal(capture[0], detector_only[0])


def test_real_decoder_exposes_actual_source_ids_after_reset():
    fixture = Path(__file__).parent / "data" / "sample.aris"
    config = YOLODatasetConfig(filepath=str(fixture), use_multithreading=False)
    _, ds = create_dataloader(config)
    with pytest.warns(UserWarning, match="Resetting start_frame"):
        frames, _ = ds.didson.load_frames(start_frame=20, end_frame=22)
    assert ds.didson.last_loaded_frame_ids == tuple(range(len(frames)))

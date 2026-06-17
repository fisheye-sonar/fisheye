import numpy as np
import pytest
import torch

from fisheye.dataloaders import (
    BaseDataset,
    create_dataloader,
    ARISBatchedDataset,
    YOLOARISBatchedDataset,
)
from fisheye.dataloaders.data_module import ARISDataModule
from fisheye.dataloaders.didson.pyDIDSON import DIDSON
from fisheye.enums import EchogramChannel
from conftest import ARIS_FILE, CORRUPTED_FILE, INVALID_FRAME_INDICES

from fisheye.configs import BaseDatasetConfig, YOLODatasetConfig


class DummyBaseDataset(BaseDataset):
    def _init_bg_frame(self):
        """Skip background precomputation for direct preprocessing tests."""

    def load_frames(self, idx, final_idx, return_unwarped=False, return_warped=True):
        raise NotImplementedError


class TestARISDataloader:
    """Test the factory function of the ARIS dataloader"""

    # Test with batch sizes smaller and larger than the number of frames
    @pytest.mark.parametrize("batch_size", [2, 32])
    def test_running_dataloader(self, batch_size):
        config = BaseDatasetConfig(filepath=ARIS_FILE, batch_size=batch_size)
        dataloader, dataset = create_dataloader(config)
        assert len(dataset) == 3
        expected_batches = max(1, (len(dataset) + batch_size - 1) // batch_size)
        assert len(dataloader) == expected_batches

        for batch in dataloader:
            batch_data, batch_labels = batch[0], batch[1]
            assert isinstance(batch_data, torch.Tensor)
            assert batch_data.dim() == 4
            assert batch_labels is None

    def test_return_unwarped_images(self):
        config = BaseDatasetConfig(filepath=ARIS_FILE, return_unwarped=True)
        dataloader, dataset = create_dataloader(config)

        batch = next(iter(dataloader))
        batch_data, batch_unwarped = batch[0], batch[2]
        assert batch_data.shape == torch.Size([3, 2686, 1307, 3])
        assert batch_unwarped.shape == torch.Size([3, 2684, 48, 3])

    def test_return_echogram(self):
        config = BaseDatasetConfig(filepath=ARIS_FILE, return_echogram=True)
        dataloader, _ = create_dataloader(config)
        _, _, batch_unwarped, batch_echogram, _ = next(iter(dataloader))
        assert batch_unwarped is None
        assert batch_echogram.shape == torch.Size([3, 2684, 3])

    def test_echogram_without_warped_frames(self):
        """Test returning only the echogram via return_frames=False."""
        config = BaseDatasetConfig(
            filepath=ARIS_FILE, return_echogram=True, return_frames=False
        )
        dataset = ARISBatchedDataset(config)
        frame_images, _, unwarped_frames, echogram, return_original_image = dataset[0]
        assert frame_images is None
        assert unwarped_frames is None
        assert return_original_image is False
        assert echogram.shape == (3, 2684, 3)

    def test_return_unwarped_without_warped_frames(self):
        config = BaseDatasetConfig(
            filepath=ARIS_FILE, return_frames=False, return_unwarped=True
        )
        dataset = ARISBatchedDataset(config)
        frame_images, _, unwarped_frames, echogram, return_original_image = dataset[0]
        assert frame_images is None
        assert unwarped_frames.shape == (3, 2684, 48, 3)
        assert echogram is None
        assert return_original_image is False

    def test_return_unwarped_without_bg_subtract_stays_single_channel(self):
        config = BaseDatasetConfig(
            filepath=ARIS_FILE,
            return_frames=False,
            return_unwarped=True,
            do_bg_subtract=False,
        )
        dataset = ARISBatchedDataset(config)
        frame_images, _, unwarped_frames, echogram, return_original_image = dataset[0]
        assert frame_images is None
        assert unwarped_frames.shape == (3, 2684, 48)
        assert echogram is None
        assert return_original_image is False

    def test_custom_echogram_channels(self):
        config = BaseDatasetConfig(
            filepath=ARIS_FILE,
            return_echogram=True,
            return_frames=False,
            echogram_channels=[EchogramChannel.BGS, EchogramChannel.ZERO, None],
        )
        dataset = ARISBatchedDataset(config)
        _, _, _, echogram, _ = dataset[0]
        assert echogram.shape == (3, 2684, 2)
        assert np.all(echogram[:, :, 1] == 0)

    def test_string_echogram_channels_are_coerced_to_enum(self):
        config = BaseDatasetConfig(filepath=ARIS_FILE, echogram_channels=["raw", "0"])
        assert config.echogram_channels == [
            EchogramChannel.RAW,
            EchogramChannel.ZERO,
        ]

    def test_only_pos_bgs_clips_negative_values(self):
        frames = np.array(
            [
                [[0, 10]],
                [[10, 20]],
                [[20, 30]],
            ],
            dtype=np.uint8,
        )
        mean_blurred_frame = np.array([[10, 10]], dtype=np.float32)
        mean_normalization_value = 10.0

        default_dataset = DummyBaseDataset(
            BaseDatasetConfig(use_blur=False, only_pos_bgs=False)
        )
        only_pos_dataset = DummyBaseDataset(
            BaseDatasetConfig(use_blur=False, only_pos_bgs=True)
        )

        default_output = default_dataset._stack_preprocessed_channels(
            frames,
            mean_blurred_frame=mean_blurred_frame,
            mean_normalization_value=mean_normalization_value,
        )
        only_pos_output = only_pos_dataset._stack_preprocessed_channels(
            frames,
            mean_blurred_frame=mean_blurred_frame,
            mean_normalization_value=mean_normalization_value,
        )

        np.testing.assert_array_equal(default_output[:, 0, :, 1], [[0, 127], [127, 255]])
        np.testing.assert_array_equal(only_pos_output[:, 0, :, 1], [[0, 0], [0, 255]])

    def test_loading_frames(self):
        """Test loading all frames directly from DIDSON class."""
        didson = DIDSON(ARIS_FILE)
        frames, unwarped_frames = didson.load_frames()
        assert isinstance(frames, np.ndarray)
        assert frames.shape == (4, 2686, 1307)  # Num of frames, ydim, xdim
        assert frames.dtype == np.uint8
        assert np.any(frames != 0)

    def test_loading_unwarped_frames(self):
        """Test loading frames directly from DIDSON class."""
        didson = DIDSON(ARIS_FILE)
        frames, unwarped_frames = didson.load_frames(return_unwarped=True)
        assert isinstance(unwarped_frames, np.ndarray)
        assert unwarped_frames.shape == (4, 2684, 48)  # Num of frames, ydim, xdim
        assert unwarped_frames.dtype == np.uint8
        assert np.any(frames != 0)

    def test_loading_unwarped_frames_without_warped(self):
        """Test return_warped=False avoids building warped images."""
        didson = DIDSON(ARIS_FILE)
        frames, unwarped_frames = didson.load_frames(
            return_unwarped=True, return_warped=False
        )
        assert frames is None
        assert unwarped_frames.shape == (4, 2684, 48)

    def test_load_echogram(self):
        didson = DIDSON(ARIS_FILE)
        echogram = didson.load_echogram(end_frame=4)
        assert echogram.shape == (3, 2684, 3)
        assert echogram.dtype == np.float32

    def test_load_echogram_matches_dataset(self):
        config = BaseDatasetConfig(
            filepath=ARIS_FILE,
            return_echogram=True,
            return_frames=False,
            use_blur=True,
        )
        dataset = ARISBatchedDataset(config)
        _, _, _, dataset_echogram, _ = dataset[0]
        didson = DIDSON(ARIS_FILE)
        didson_echogram = didson.load_echogram(end_frame=4)
        np.testing.assert_allclose(dataset_echogram, didson_echogram, rtol=1e-5)

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 4, 3),  # Good! In-range
            (0, 2, 1),  # Good! In-range
            (6, 4, 3),  # Bad range: start > end
            (0, 6, 3),  # end_frame out of range
            (6, 5, 3),  # start_frame > file length
        ],
    )
    def test_loading_different_frame_ranges(
        self, start_frame, end_frame, expected_length
    ):
        """Test ARIS factory function correctly handles frame range validation for ARIS datasets."""
        config = BaseDatasetConfig(
            filepath=ARIS_FILE, start_frame=start_frame, end_frame=end_frame
        )
        dataloader, dataset = create_dataloader(config)
        assert len(dataset) == expected_length

    def test_requires_at_least_one_output(self):
        with pytest.raises(ValueError, match="At least one of return_frames"):
            ARISBatchedDataset(
                BaseDatasetConfig(
                    filepath=ARIS_FILE,
                    return_frames=False,
                    return_unwarped=False,
                    return_echogram=False,
                )
            )


class TestYOLODataloader:
    """Test the factory function of the YOLOv5 dataloader"""

    @pytest.mark.parametrize("batch_size", [2, 32])
    def test_running_dataloader(self, batch_size):
        """Test creating a YOLO dataloader using factory function with no labels."""
        didson = DIDSON(ARIS_FILE)
        frames, unwarped_frames = didson.load_frames()

        config = YOLODatasetConfig(filepath=ARIS_FILE, batch_size=batch_size)
        dataloader, dataset = create_dataloader(config)
        num_batches = len(dataloader)

        assert len(dataset) == len(frames) - 1
        expected_batches = max(1, (len(dataset) + batch_size - 1) // batch_size)
        assert num_batches == expected_batches

        for batch in dataloader:
            batch_data, batch_labels = batch[0], batch[1]
            assert isinstance(batch_data, torch.Tensor)
            assert batch_data.dim() == 4
            assert batch_labels is None or batch_labels.numel() == 0

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 4, 3),  # Good! In-range
            (0, 2, 1),  # Good! In-range
            (6, 4, 3),  # Bad range: start > end
            (0, 6, 3),  # end_frame out of range
            (6, 5, 3),  # start_frame > file length
        ],
    )
    def test_loading_different_frame_ranges(
        self, start_frame, end_frame, expected_length
    ):
        """Test YOLO factory function correctly handles frame range validation for YOLO datasets."""
        config = YOLODatasetConfig(
            filepath=ARIS_FILE, start_frame=start_frame, end_frame=end_frame
        )
        dataloader, dataset = create_dataloader(config)
        assert len(dataset) == expected_length


class TestARISLightningDataloader:
    """Test the ARIS Lightning Datamodule"""

    def test_running_dataloader(self):
        """Test creating a ARIS dataloader using Lightning DataModule with no labels."""
        config = BaseDatasetConfig(filepath=ARIS_FILE)
        data_module = ARISDataModule(ARISBatchedDataset, config)
        data_module.setup(stage="test")
        dataloader = data_module.test_dataloader()

        num_batches = len(dataloader)
        assert num_batches == 1

        batch = next(iter(dataloader))
        batch_data, batch_labels = batch[0], batch[1]

        # Check batch content
        assert batch_data.shape == torch.Size([3, 2686, 1307, 3])
        # Check batch labels
        assert batch_labels is None

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 4, 3),  # Good! In-range
            (0, 2, 1),  # Good! In-range
            (6, 4, 3),  # Bad range: start > end
            (0, 6, 3),  # end_frame out of range
            (6, 5, 3),  # start_frame > file length
        ],
    )
    def test_loading_different_frame_ranges(
        self, start_frame, end_frame, expected_length
    ):
        """Test ARIS DataModule correctly handles frame range validation for ARIS datasets."""
        config = BaseDatasetConfig(
            filepath=ARIS_FILE, start_frame=start_frame, end_frame=end_frame
        )
        data_module = ARISDataModule(ARISBatchedDataset, config)
        data_module.setup(stage="test")

        assert len(data_module.dataset) == expected_length


class TestYOLOLightningDataloader:
    def test_running_dataloader(self):
        """Test creating a YOLO dataloader using Lightning DataModule with no labels."""
        config = YOLODatasetConfig(filepath=ARIS_FILE)
        data_module = ARISDataModule(YOLOARISBatchedDataset, config)
        data_module.setup(stage="test")
        dataloader = data_module.test_dataloader()

        num_batches = len(dataloader)
        assert num_batches == 1

        batch = next(iter(dataloader))
        # Check batch content
        assert batch[0].shape == torch.Size([3, 3, 960, 512])
        # Check batch labels
        assert batch[1] is None or batch[1].numel() == 0

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 4, 3),  # Good! In-range
            (0, 2, 1),  # Good! In-range
            (6, 4, 3),  # Bad range: start > end
            (0, 6, 3),  # end_frame out of range
            (6, 5, 3),  # start_frame > file length
        ],
    )
    def test_loading_different_frame_ranges(
        self, start_frame, end_frame, expected_length
    ):
        """Test ARIS DataModule correctly handles frame range validation for YOLO datasets."""
        config = YOLODatasetConfig(
            filepath=ARIS_FILE, start_frame=start_frame, end_frame=end_frame
        )
        data_module = ARISDataModule(YOLOARISBatchedDataset, config)
        data_module.setup(stage="test")

        assert len(data_module.dataset) == expected_length


def test_loading_unwarped_frames_from_didson(beam_widths_path):
    """Test loading frames directly from DIDSON class."""
    didson = DIDSON(ARIS_FILE, beam_widths_path)
    frames, unwarped_frames = didson.load_frames(return_unwarped=True)
    assert isinstance(unwarped_frames, np.ndarray)
    assert unwarped_frames.shape == (4, 2684, 48)  # Num of frames, ydim, xdim
    assert unwarped_frames.dtype == np.uint8
    assert np.any(frames != 0)


def test_corrupted_file():
    """Test dataloader fails to create due to corrupted ARIS file."""
    with pytest.raises(RuntimeError) as exc_info:
        config = BaseDatasetConfig(filepath=CORRUPTED_FILE)
        create_dataloader(config)


def test_reset_start_end_frames_when_exceeding_total_frames():
    """Test start and end frames reset if they are larger than the total number of frames in the file."""
    # Start frame in header is 100 and end frame is 110
    config = BaseDatasetConfig(filepath=INVALID_FRAME_INDICES)
    dataloader, dataset = create_dataloader(config)

    # originally 10 frames in file, but subtract 1 for optical flow
    assert len(dataset) == 9

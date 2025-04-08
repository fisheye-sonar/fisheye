import numpy as np
import pytest
import torch

from fisheye.dataloaders import (
    create_aris_dataloader,
    ARISBatchedDataset,
    YOLOARISBatchedDataset,
)
from fisheye.dataloaders.data_module import ARISDataModule
from fisheye.dataloaders.didson.pyDIDSON import DIDSON
from conftest import ARIS_FILE, CORRUPTED_FILE, INVALID_FRAME_INDICES
from fisheye.dataloaders.yolo import create_yolo_dataloader

from fisheye.configs import ARISDatasetConfig, YOLODatasetConfig


class TestARISDataloader:
    """Test the factory function of the ARIS dataloader"""

    # Test with batch sizes smaller and larger than the number of frames
    @pytest.mark.parametrize("batch_size", [2, 32])
    def test_running_dataloader(self, batch_size):
        config = ARISDatasetConfig(filepath=ARIS_FILE, batch_size=batch_size)
        dataloader, dataset = create_aris_dataloader(config)
        assert len(dataset) == 3
        expected_batches = max(1, (len(dataset) + batch_size - 1) // batch_size)
        assert len(dataloader) == expected_batches

        for batch in dataloader:
            batch_data, batch_labels = batch[0], batch[1]
            assert isinstance(batch_data, torch.Tensor)
            assert batch_data.dim() == 4
            assert batch_labels is None

    def test_return_unwarped_images(self):
        config = ARISDatasetConfig(filepath=ARIS_FILE, return_unwarped=True)
        dataloader, dataset = create_aris_dataloader(config)

        batch = next(iter(dataloader))
        batch_data, batch_unwarped = batch[0], batch[2]
        assert batch_data.shape == torch.Size([3, 2684, 48, 3])
        assert batch_unwarped.shape == torch.Size([3, 2684, 48])

    def test_return_echogram(self):
        config = ARISDatasetConfig(filepath=ARIS_FILE, return_echogram=True)
        dataloader, _ = create_aris_dataloader(config)
        _, _, _, batch_echogram = next(iter(dataloader))
        assert batch_echogram.shape == torch.Size([3, 2684, 2])

    def test_loading_frames(self):
        """Test loading frames directly from DIDSON class."""
        didson = DIDSON(ARIS_FILE)
        frames, unwarped_frames = didson.load_frames()
        assert isinstance(frames, np.ndarray)
        assert frames.shape == (4, 2686, 1307)  # Num of frames, ydim, xdim
        assert frames.dtype == np.uint8
        assert np.any(frames != 0)

    def test_loading_selected_frames(self):
        """Test ARIS factory function correctly loads frames from specified range."""
        config = ARISDatasetConfig(filepath=ARIS_FILE)
        dataloader, dataset = create_aris_dataloader(config)

        # originally 4 frames in file, but subtract 1 for optical flow
        assert len(dataset) == 3

        config = ARISDatasetConfig(filepath=ARIS_FILE, start_frame=0, end_frame=2)
        dataloader, dataset = create_aris_dataloader(config)

        # end_frame is exclusive in DIDSON
        assert len(dataset) == 1

        config = ARISDatasetConfig(filepath=ARIS_FILE, start_frame=1, end_frame=4)
        dataloader, dataset = create_aris_dataloader(config)

        # end_frame is exclusive in DIDSON
        assert len(dataset) == 2

    def test_loading_unwarped_frames(self):
        """Test loading frames directly from DIDSON class."""
        didson = DIDSON(ARIS_FILE)
        frames, unwarped_frames = didson.load_frames(return_unwarped=True)
        assert isinstance(unwarped_frames, np.ndarray)
        assert unwarped_frames.shape == (4, 2684, 48)  # Num of frames, ydim, xdim
        assert unwarped_frames.dtype == np.uint8
        assert np.any(frames != 0)

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 4, 3),  # Good!
            (6, 4, 3),  # Bad range: start > end
            (0, 6, 3),  # end_frame out of range
            (6, 5, 3),  # start_frame > file length
        ],
    )
    def test_loading_bad_frame_ranges(self, start_frame, end_frame, expected_length):
        """Test ARIS factory function does not load frames from an outside range."""
        config = ARISDatasetConfig(
            filepath=ARIS_FILE, start_frame=start_frame, end_frame=end_frame
        )
        dataloader, dataset = create_aris_dataloader(config)
        assert len(dataset) == expected_length


class TestYOLODataloader:
    """Test the factory function of the YOLOv5 dataloader"""

    @pytest.mark.parametrize("batch_size", [2, 32])
    def test_running_dataloader(self, batch_size):
        """Test creating a YOLO dataloader using factory function with no labels."""
        didson = DIDSON(ARIS_FILE)
        frames, unwarped_frames = didson.load_frames()

        config = YOLODatasetConfig(filepath=ARIS_FILE, batch_size=batch_size)
        dataloader, dataset = create_yolo_dataloader(config)
        num_batches = len(dataloader)

        assert len(dataset) == len(frames) - 1
        expected_batches = max(1, (len(dataset) + batch_size - 1) // batch_size)
        assert len(dataloader) == expected_batches

        for batch in dataloader:
            batch_data, batch_labels = batch[0], batch[1]
            assert isinstance(batch_data, torch.Tensor)
            assert batch_data.dim() == 4
            assert batch_labels is None or batch_labels.numel() == 0

    def test_loading_selected_frames(self):
        """Test YOLO factory function correctly loads frames from specified range."""
        config = YOLODatasetConfig(filepath=ARIS_FILE)
        dataloader, dataset = create_yolo_dataloader(config)

        # originally 4 frames in file, but subtract 1 for optical flow
        assert len(dataset) == 3

        config = YOLODatasetConfig(filepath=ARIS_FILE, start_frame=0, end_frame=2)
        dataloader, dataset = create_yolo_dataloader(config)

        # end_frame is exclusive in DIDSON
        assert len(dataset) == 1

        config = YOLODatasetConfig(filepath=ARIS_FILE, start_frame=1, end_frame=4)
        dataloader, dataset = create_yolo_dataloader(config)

        assert len(dataset) == 2

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 4, 3),  # Good!
            (6, 4, 3),  # Bad range: start > end
            (0, 6, 3),  # end_frame out of range
            (6, 5, 3),  # start_frame > file length
        ],
    )
    def test_loading_bad_frame_ranges(self, start_frame, end_frame, expected_length):
        """Test YOLO factory function does not load frames from an outside range."""
        config = YOLODatasetConfig(
            filepath=ARIS_FILE, start_frame=start_frame, end_frame=end_frame
        )
        dataloader, dataset = create_yolo_dataloader(config)
        assert len(dataset) == expected_length


class TestARISLightningDataloader:
    """Test the ARIS Lightning Datamodule"""

    def test_running_dataloader(self):
        """Test creating a ARIS dataloader using Lightning DataModule with no labels."""
        config = ARISDatasetConfig(filepath=ARIS_FILE)
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

    def test_loading_selected_frames(self):
        """Test ARIS DataModule correctly loads frames from specified range."""
        config = ARISDatasetConfig(filepath=ARIS_FILE)
        data_module = ARISDataModule(ARISBatchedDataset, config)
        data_module.setup(stage="test")

        # originally 4 frames in file, but subtract 1 for optical flow
        assert len(data_module.dataset) == 3

        config = ARISDatasetConfig(filepath=ARIS_FILE, start_frame=0, end_frame=2)
        data_module = ARISDataModule(ARISBatchedDataset, config)
        data_module.setup(stage="test")

        # end_frame is exclusive in DIDSON
        assert len(data_module.dataset) == 1

        config = ARISDatasetConfig(filepath=ARIS_FILE, start_frame=1, end_frame=4)
        data_module = ARISDataModule(ARISBatchedDataset, config)
        data_module.setup(stage="test")

        # end_frame is exclusive in DIDSON
        assert len(data_module.dataset) == 2

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 4, 3),  # Good!
            (6, 4, 3),  # Bad range: start > end
            (0, 6, 3),  # end_frame out of range
            (6, 5, 3),  # start_frame > file length
        ],
    )
    def test_loading_bad_frame_ranges(self, start_frame, end_frame, expected_length):
        """Test ARIS DataModule does not load frames from an outside range for ARIS Datasets."""
        config = ARISDatasetConfig(
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

    def test_loading_selected_frames(self):
        """Test ARIS DataModule correctly loads frames from specified range for YOLO Datasets."""
        config = YOLODatasetConfig(filepath=ARIS_FILE)
        data_module = ARISDataModule(YOLOARISBatchedDataset, config)
        data_module.setup(stage="test")

        # originally 4 frames in file, but subtract 1 for optical flow
        assert len(data_module.dataset) == 3

        config = YOLODatasetConfig(filepath=ARIS_FILE, start_frame=0, end_frame=2)
        data_module = ARISDataModule(YOLOARISBatchedDataset, config)
        data_module.setup(stage="test")
        assert len(data_module.dataset) == 1

        config = YOLODatasetConfig(filepath=ARIS_FILE, start_frame=1, end_frame=4)
        data_module = ARISDataModule(YOLOARISBatchedDataset, config)
        data_module.setup(stage="test")
        assert len(data_module.dataset) == 2

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 4, 3),  # Good!
            (6, 4, 3),  # Bad range: start > end
            (0, 6, 3),  # end_frame out of range
            (6, 5, 3),  # start_frame > file length
        ],
    )
    def test_loading_bad_frame_ranges(self, start_frame, end_frame, expected_length):
        """Test ARIS DataModule does not load frames from an outside range for ARIS Datasets."""
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
        config = ARISDatasetConfig(filepath=CORRUPTED_FILE)
        create_aris_dataloader(config)


def test_reset_start_end_frames_when_exceeding_total_frames():
    """Test start and end frames reset if they are larger than the total number of frames in the file."""
    # Start frame in header is 100 and end frame is 110
    config = ARISDatasetConfig(filepath=INVALID_FRAME_INDICES)
    dataloader, dataset = create_aris_dataloader(config)

    # originally 10 frames in file, but subtract 1 for optical flow
    assert len(dataset) == 9

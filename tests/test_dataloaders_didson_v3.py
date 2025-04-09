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
from conftest import CORRUPTED_FILE, DDF_FILE, SHORTENED_DDF_FILE
from fisheye.dataloaders.yolo import create_yolo_dataloader

from fisheye.configs import ARISDatasetConfig, YOLODatasetConfig

"""The same as test_dataloaders but now running on a didson version 3 file NOTE: Currently the warped images are 
being returned as [486, 300], this is smaller than the unwarped images [512, 96] when it should be at least the same 
height. This means we are losing resolution in the warp."""


class TestARISDataloader:
    """Test the factory function of the ARIS dataloader"""

    # Test with batch sizes smaller and larger than the number of frames
    @pytest.mark.parametrize("batch_size", [2, 32])
    def test_running_dataloader(self, batch_size):
        config = ARISDatasetConfig(filepath=DDF_FILE, batch_size=batch_size)
        dataloader, dataset = create_aris_dataloader(config)
        assert len(dataset) == 133
        expected_batches = max(1, (len(dataset) + batch_size - 1) // batch_size)
        assert len(dataloader) == expected_batches

        for batch in dataloader:
            batch_data, batch_labels = batch[0], batch[1]
            assert isinstance(batch_data, torch.Tensor)
            assert batch_data.dim() == 4
            assert batch_labels is None

    def test_return_unwarped_images(self):
        config = ARISDatasetConfig(filepath=DDF_FILE, return_unwarped=True)
        dataloader, dataset = create_aris_dataloader(config)

        batch = next(iter(dataloader))
        batch_data, batch_unwarped = batch[0], batch[2]
        assert batch_data.shape == torch.Size([32, 512, 96, 3])
        assert batch_unwarped.shape == torch.Size([32, 512, 96])

    def test_return_echogram(self):
        config = ARISDatasetConfig(filepath=DDF_FILE, return_echogram=True)
        dataloader, _ = create_aris_dataloader(config)
        _, _, _, batch_echogram = next(iter(dataloader))
        assert batch_echogram.shape == torch.Size([32, 512, 2])

    def test_loading_frames(self):
        """Test loading frames directly from DIDSON class."""
        didson = DIDSON(DDF_FILE)
        frames, unwarped_frames = didson.load_frames()
        assert isinstance(frames, np.ndarray)
        assert frames.shape == (134, 486, 300)  # Num of frames, ydim, xdim
        assert frames.dtype == np.uint8
        assert np.any(frames != 0)

    def test_loading_selected_frames(self):
        """Test ARIS factory function correctly loads frames from specified range."""
        config = ARISDatasetConfig(filepath=DDF_FILE)
        dataloader, dataset = create_aris_dataloader(config)
        # originally 4 frames in file, but subtract 1 for optical flow
        assert len(dataset) == 133

        config = ARISDatasetConfig(filepath=DDF_FILE, start_frame=0, end_frame=2)
        dataloader, dataset = create_aris_dataloader(config)
        # end_frame is exclusive in DIDSON
        assert len(dataset) == 1

        config = ARISDatasetConfig(filepath=DDF_FILE, start_frame=120, end_frame=135)
        dataloader, dataset = create_aris_dataloader(config)
        # end_frame is exclusive in DIDSON
        assert len(dataset) == 13

    def test_loading_unwarped_frames(self):
        """Test loading frames directly from DIDSON class."""
        didson = DIDSON(DDF_FILE)
        frames, unwarped_frames = didson.load_frames(return_unwarped=True)
        assert isinstance(unwarped_frames, np.ndarray)
        assert unwarped_frames.shape == (134, 512, 96)  # Num of frames, ydim, xdim
        assert unwarped_frames.dtype == np.uint8
        assert np.any(frames != 0)

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 134, 133),  # Good! In-range
            (0, 2, 1),  # Good! In-range
            (120, 140, 13),  # end_frame out of range
            (300, 134, 133),  # start_frame > file length
        ],
    )
    def test_loading_bad_frame_ranges(self, start_frame, end_frame, expected_length):
        """Test ARIS factory function does not load frames from an outside range."""
        config = ARISDatasetConfig(
            filepath=DDF_FILE, start_frame=start_frame, end_frame=end_frame
        )
        dataloader, dataset = create_aris_dataloader(config)
        assert len(dataset) == expected_length


class TestARISLightningDataloader:
    """Test the ARIS Lightning Datamodule"""

    @pytest.mark.parametrize("batch_size", [2, 32])
    def test_running_dataloader(self, batch_size):
        """Test creating a ARIS dataloader using Lightning DataModule with no labels."""
        config = ARISDatasetConfig(filepath=DDF_FILE, batch_size=batch_size)
        data_module = ARISDataModule(ARISBatchedDataset, config)
        data_module.setup(stage="test")
        dataloader = data_module.test_dataloader()

        num_batches = len(dataloader)
        expected_batches = max(
            1, (len(dataloader.dataset) + batch_size - 1) // batch_size
        )
        assert num_batches == expected_batches

        batch = next(iter(dataloader))
        batch_data, batch_labels = batch[0], batch[1]

        # Check batch content
        assert batch_data.shape == torch.Size([batch_size, 486, 300, 3])
        # Check batch labels
        assert batch_labels is None or batch_labels.numel() == 0

    def test_loading_selected_frames(self):
        """Test ARIS DataModule correctly loads frames from specified range."""
        config = ARISDatasetConfig(filepath=DDF_FILE)
        data_module = ARISDataModule(ARISBatchedDataset, config)
        data_module.setup(stage="test")

        # originally 134 frames in file, but subtract 1 for optical flow
        assert len(data_module.dataset) == 133

        config = ARISDatasetConfig(filepath=DDF_FILE, start_frame=0, end_frame=2)
        data_module = ARISDataModule(ARISBatchedDataset, config)
        data_module.setup(stage="test")

        # end_frame is exclusive in DIDSON
        assert len(data_module.dataset) == 1

        config = ARISDatasetConfig(filepath=DDF_FILE, start_frame=120, end_frame=135)
        data_module = ARISDataModule(ARISBatchedDataset, config)
        data_module.setup(stage="test")

        # end_frame is exclusive in DIDSON
        assert len(data_module.dataset) == 13

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 134, 133),  # Good! In-range
            (0, 2, 1),  # Good! In-range
            (120, 140, 13),  # end_frame out of range
            (300, 134, 133),  # start_frame > file length
        ],
    )
    def test_loading_bad_frame_ranges(self, start_frame, end_frame, expected_length):
        """Test ARIS DataModule does not load frames from an outside range for ARIS Datasets."""
        config = ARISDatasetConfig(
            filepath=DDF_FILE, start_frame=start_frame, end_frame=end_frame
        )
        data_module = ARISDataModule(ARISBatchedDataset, config)
        data_module.setup(stage="test")

        assert len(data_module.dataset) == expected_length


class TestYOLODataloader:
    """Test the factory function of the YOLOv5 dataloader"""

    @pytest.mark.parametrize("batch_size", [2, 32])
    def test_running_dataloader(self, batch_size):
        """Test creating a YOLO dataloader using factory function with no labels."""
        didson = DIDSON(DDF_FILE)
        frames, unwarped_frames = didson.load_frames()

        config = YOLODatasetConfig(filepath=DDF_FILE, batch_size=batch_size)
        dataloader, dataset = create_yolo_dataloader(config)
        num_batches = len(dataloader)

        assert len(dataset) == len(frames) - 1
        expected_batches = max(1, (len(dataset) + batch_size - 1) // batch_size)
        assert num_batches == expected_batches

        for batch in dataloader:
            batch_data, batch_labels = batch[0], batch[1]
            assert isinstance(batch_data, torch.Tensor)
            assert batch_data.dim() == 4
            assert batch_data[0].shape == torch.Size([3, 960, 640])
            assert batch_labels is None or batch_labels.numel() == 0

    def test_loading_selected_frames(self):
        """Test YOLO factory function correctly loads frames from specified range."""
        config = YOLODatasetConfig(filepath=DDF_FILE)
        dataloader, dataset = create_yolo_dataloader(config)

        # originally 4 frames in file, but subtract 1 for optical flow
        assert len(dataset) == 133

        config = YOLODatasetConfig(filepath=DDF_FILE, start_frame=0, end_frame=2)
        dataloader, dataset = create_yolo_dataloader(config)

        # end_frame is exclusive in DIDSON
        assert len(dataset) == 1

        config = YOLODatasetConfig(filepath=DDF_FILE, start_frame=120, end_frame=135)
        dataloader, dataset = create_yolo_dataloader(config)

        # end_frame is exclusive in DIDSON
        assert len(dataset) == 13

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 134, 133),  # Good! In-range
            (0, 2, 1),  # Good! In-range
            (120, 140, 13),  # end_frame out of range
            (300, 134, 133),  # start_frame > file length
        ],
    )
    def test_loading_bad_frame_ranges(self, start_frame, end_frame, expected_length):
        """Test YOLO factory function does not load frames from an outside range."""
        config = YOLODatasetConfig(
            filepath=DDF_FILE, start_frame=start_frame, end_frame=end_frame
        )
        dataloader, dataset = create_yolo_dataloader(config)
        assert len(dataset) == expected_length


class TestYOLOLightningDataloader:
    @pytest.mark.parametrize("batch_size", [2, 32])
    def test_running_dataloader(self, batch_size):
        """Test creating a YOLO dataloader using Lightning DataModule with no labels."""
        config = YOLODatasetConfig(filepath=DDF_FILE, batch_size=batch_size)
        data_module = ARISDataModule(YOLOARISBatchedDataset, config)
        data_module.setup(stage="test")
        dataloader = data_module.test_dataloader()

        num_batches = len(dataloader)
        expected_batches = max(
            1, (len(dataloader.dataset) + batch_size - 1) // batch_size
        )
        assert num_batches == expected_batches

        batch = next(iter(dataloader))
        batch_data, batch_labels = batch[0], batch[1]

        # Check batch content
        assert batch_data.shape == torch.Size([batch_size, 3, 960, 640])
        # Check batch labels
        assert batch_labels is None or batch_labels.numel() == 0

    def test_loading_selected_frames(self):
        """Test ARIS DataModule correctly loads frames from specified range for YOLO Datasets."""
        config = YOLODatasetConfig(filepath=DDF_FILE)
        data_module = ARISDataModule(YOLOARISBatchedDataset, config)
        data_module.setup(stage="test")

        # originally 4 frames in file, but subtract 1 for optical flow
        assert len(data_module.dataset) == 133

        config = YOLODatasetConfig(filepath=DDF_FILE, start_frame=0, end_frame=2)
        data_module = ARISDataModule(YOLOARISBatchedDataset, config)
        data_module.setup(stage="test")
        assert len(data_module.dataset) == 1

        config = YOLODatasetConfig(filepath=DDF_FILE, start_frame=120, end_frame=135)
        data_module = ARISDataModule(YOLOARISBatchedDataset, config)
        data_module.setup(stage="test")
        assert len(data_module.dataset) == 13

    @pytest.mark.parametrize(
        "start_frame, end_frame, expected_length",
        [
            (0, 134, 133),  # Good! In-range
            (0, 2, 1),  # Good! In-range
            (120, 140, 13),  # end_frame out of range
            (300, 134, 133),  # start_frame > file length
        ],
    )
    def test_loading_bad_frame_ranges(self, start_frame, end_frame, expected_length):
        """Test ARIS DataModule does not load frames from an outside range for ARIS Datasets."""
        config = YOLODatasetConfig(
            filepath=DDF_FILE, start_frame=start_frame, end_frame=end_frame
        )
        data_module = ARISDataModule(YOLOARISBatchedDataset, config)
        data_module.setup(stage="test")

        assert len(data_module.dataset) == expected_length


def test_corrupted_aris():
    """Test dataloader fails to create due to corrupted ARIS file."""
    with pytest.raises(RuntimeError) as exc_info:
        config = ARISDatasetConfig(filepath=CORRUPTED_FILE)
        create_aris_dataloader(config)


def test_modified_start_end_frames(beam_widths_path):
    """Test handling modified start and end frame indices exceeding the total number of frames. This is a test case
    for when a shorted clip is created from the original ARIS/DIDSON file."""
    didson = DIDSON(SHORTENED_DDF_FILE, beam_widths_path)
    frames, unwarped_frames = didson.load_frames()

    assert isinstance(frames, np.ndarray)
    assert frames.shape == (57, 486, 300)  # Num of frames, ydim, xdim
    assert frames.dtype == np.uint8
    assert np.any(frames != 0)

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
from conftest import ARIS_FILE, CORRUPTED_FILE
from fisheye.dataloaders.yolo import create_yolo_dataloader

from fisheye.config import ARISDatasetConfig, YOLODatasetConfig


def test_creating_aris_dataloader_factory_func(beam_widths_path):
    """Test creating a ARIS dataloader using factory function with no labels."""
    didson = DIDSON(ARIS_FILE, beam_widths_path)
    frames, unwarped_frames = didson.load_frames()

    config = ARISDatasetConfig(aris_filepath=ARIS_FILE)
    dataloader, dataset = create_aris_dataloader(config)
    num_batches = len(dataloader)
    assert len(dataset) == len(frames) - 1
    assert num_batches == 1

    batch = next(iter(dataloader))
    batch_data, batch_labels = batch[0], batch[1]
    # Check batch content
    assert batch_data.shape == torch.Size([3, 2686, 1307, 3])
    # Check batch labels
    assert batch_labels is None


def test_creating_aris_dataloader_lightning(beam_widths_path):
    """Test creating a ARIS dataloader using Lightning DataModule with no labels."""
    config = ARISDatasetConfig(aris_filepath=ARIS_FILE)
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


def test_creating_yolo_dataloader_factory_func(beam_widths_path):
    """Test creating a YOLO dataloader using factory function with no labels."""
    didson = DIDSON(ARIS_FILE, beam_widths_path)
    frames, unwarped_frames = didson.load_frames()

    config = YOLODatasetConfig(aris_filepath=ARIS_FILE)
    dataloader, dataset = create_yolo_dataloader(config)
    num_batches = len(dataloader)

    assert len(dataset) == len(frames) - 1
    assert num_batches == 1

    batch = next(iter(dataloader))
    # Check batch content
    assert batch[0].shape == torch.Size([3, 3, 960, 512])
    # Check batch labels
    assert batch[1] is None or batch[1].numel() == 0


def test_creating_yolo_dataloader_lightning(beam_widths_path):
    """Test creating a YOLO dataloader using Lightning DataModule with no labels."""
    config = YOLODatasetConfig(aris_filepath=ARIS_FILE)
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


def test_aris_loading_frames(beam_widths_path):
    """Test loading frames directly from DIDSON class."""
    didson = DIDSON(ARIS_FILE, beam_widths_path)
    frames, unwarped_frames = didson.load_frames()
    assert isinstance(frames, np.ndarray)
    assert isinstance(unwarped_frames, np.ndarray)
    assert frames.shape == (4, 2686, 1307)  # Num of frames, ydim, xdim
    assert unwarped_frames.shape == (4, 2684, 48)  # Num of frames, ydim, xdim
    assert frames.dtype == np.uint8
    assert unwarped_frames.dtype == np.uint8


def test_loading_selected_frames_aris_dataloader_factory_func():
    """Test ARIS factory function correctly loads frames from specified range."""
    config = ARISDatasetConfig(aris_filepath=ARIS_FILE)
    dataloader, dataset = create_aris_dataloader(config)

    # originally 4 frames in file, but subtract 1 for optical flow
    assert len(dataset) == 3

    config = ARISDatasetConfig(aris_filepath=ARIS_FILE, start_frame=0, end_frame=2)
    dataloader, dataset = create_aris_dataloader(config)

    # end_frame is exclusive in DIDSON
    assert len(dataset) == 1


def test_loading_bad_end_frame_aris_dataloader_factory_func():
    """Test ARIS factory function does not load frames from an outside range."""
    config = ARISDatasetConfig(aris_filepath=ARIS_FILE)
    dataloader, dataset = create_aris_dataloader(config)

    # originally 4 frames in file, but subtract 1 for optical flow
    assert len(dataset) == 3

    config = ARISDatasetConfig(aris_filepath=ARIS_FILE, start_frame=0, end_frame=6)
    dataloader, dataset = create_aris_dataloader(config)

    # Defaults to using end frame from file header if end_frame specified is larger and doesn't exist
    assert len(dataset) == 3


def test_loading_selected_frames_aris_dataloader_lightning():
    """Test ARIS DataModule correctly loads frames from specified range."""
    config = ARISDatasetConfig(aris_filepath=ARIS_FILE)
    data_module = ARISDataModule(ARISBatchedDataset, config)
    data_module.setup(stage="test")

    # originally 4 frames in file, but subtract 1 for optical flow
    assert len(data_module.dataset) == 3

    config = ARISDatasetConfig(aris_filepath=ARIS_FILE, start_frame=0, end_frame=2)
    data_module = ARISDataModule(ARISBatchedDataset, config)
    data_module.setup(stage="test")

    # end_frame is exclusive in DIDSON
    assert len(data_module.dataset) == 1


def test_loading_bad_end_frame_aris_dataloader_lightning():
    """Test ARIS DataModule does not load frames from an outside range."""
    config = ARISDatasetConfig(aris_filepath=ARIS_FILE)
    data_module = ARISDataModule(ARISBatchedDataset, config)
    data_module.setup(stage="test")

    # originally 4 frames in file, but subtract 1 for optical flow
    assert len(data_module.dataset) == 3

    config = ARISDatasetConfig(aris_filepath=ARIS_FILE, start_frame=0, end_frame=6)
    data_module = ARISDataModule(ARISBatchedDataset, config)
    data_module.setup(stage="test")

    # Defaults to using end frame - 1 from file header if end_frame specified is larger and doesn't exist
    assert len(data_module.dataset) == 3


def test_loading_selected_frames_yolo_dataloader_factory_func():
    """Test YOLO factory function correctly loads frames from specified range."""
    config = YOLODatasetConfig(aris_filepath=ARIS_FILE)
    dataloader, dataset = create_yolo_dataloader(config)

    # originally 4 frames in file, but subtract 1 for optical flow
    assert len(dataset) == 3

    config = YOLODatasetConfig(aris_filepath=ARIS_FILE, start_frame=0, end_frame=2)
    dataloader, dataset = create_yolo_dataloader(config)

    # end_frame is exclusive in DIDSON
    assert len(dataset) == 1


def test_loading_bad_end_frame_yolo_dataloader_factory_func():
    """Test YOLO factory function does not load frames from an outside range."""
    config = YOLODatasetConfig(aris_filepath=ARIS_FILE)
    dataloader, dataset = create_yolo_dataloader(config)

    # originally 4 frames in file, but subtract 1 for optical flow
    assert len(dataset) == 3

    config = YOLODatasetConfig(aris_filepath=ARIS_FILE, start_frame=0, end_frame=6)
    dataloader, dataset = create_yolo_dataloader(config)

    # end_frame is exclusive in DIDSON
    assert len(dataset) == 3


def test_loading_selected_frames_yolo_dataloader_lightning():
    """Test ARIS DataModule correctly loads frames from specified range for YOLO Datasets."""
    config = YOLODatasetConfig(aris_filepath=ARIS_FILE)
    data_module = ARISDataModule(YOLOARISBatchedDataset, config)
    data_module.setup(stage="test")

    # originally 4 frames in file, but subtract 1 for optical flow
    assert len(data_module.dataset) == 3

    config = YOLODatasetConfig(aris_filepath=ARIS_FILE, start_frame=0, end_frame=2)
    data_module = ARISDataModule(YOLOARISBatchedDataset, config)
    data_module.setup(stage="test")
    assert len(data_module.dataset) == 1


def test_loading_bad_end_frame_yolo_dataloader_lightning():
    """Test ARIS DataModule does not load frames from an outside range for YOLO Datasets."""
    config = YOLODatasetConfig(aris_filepath=ARIS_FILE)
    data_module = ARISDataModule(YOLOARISBatchedDataset, config)
    data_module.setup(stage="test")

    # originally 4 frames in file, but subtract 1 for optical flow
    assert len(data_module.dataset) == 3

    config = YOLODatasetConfig(aris_filepath=ARIS_FILE, start_frame=0, end_frame=6)
    data_module = ARISDataModule(YOLOARISBatchedDataset, config)
    data_module.setup(stage="test")
    # Defaults to using end frame from file header if end_frame specified is larger and doesn't exist
    assert len(data_module.dataset) == 3


def test_corrupted_aris():
    """Test dataloader fails to create due to corrupted ARIS file."""
    with pytest.raises(RuntimeError) as exc_info:
        config = ARISDatasetConfig(aris_filepath=CORRUPTED_FILE)
        create_aris_dataloader(config)

import numpy as np
import pytest
import torch

from fisheye.dataloaders import create_aris_dataloader, ARISBatchedDataset, YOLOARISBatchedDataset
from fisheye.dataloaders.data_module import ARISDataModule
from fisheye.dataloaders.didson.pyDIDSON import DIDSON
from conftest import ARIS_FILE, CORRUPTED_FILE
from fisheye.dataloaders.yolo import create_yolo_dataloader


def test_aris_loading_frames(beam_widths_path):
    didson = DIDSON(ARIS_FILE, beam_widths_path)
    frames = didson.load_frames()
    assert isinstance(frames, np.ndarray)
    assert frames.shape == (4, 2686, 1307)  # Num of frames, ydim, xdim
    assert frames.dtype == np.uint8


def test_corrupted_aris():
    with pytest.raises(RuntimeError) as exc_info:
        create_aris_dataloader(CORRUPTED_FILE)


def test_creating_aris_dataloader_factory_func(beam_widths_path):
    """Test creating a ARIS dataloader using factory function with no labels."""
    didson = DIDSON(ARIS_FILE, beam_widths_path)
    frames = didson.load_frames()

    dataloader, dataset = create_aris_dataloader(ARIS_FILE)
    num_batches = len(dataloader)

    assert len(dataset) == len(frames) - 1
    assert num_batches == 1

    batch = next(iter(dataloader))
    batch_data, batch_labels = batch
    # Check batch content
    assert batch_data.shape == torch.Size([3, 2686, 1307, 3])
    # Check batch labels
    assert batch_labels is None


def test_creating_aris_dataloader_lightning(beam_widths_path):
    """Test creating a ARIS dataloader using Lightning DataModule with no labels."""
    data_module = ARISDataModule(ARISBatchedDataset, {'aris_filepath': ARIS_FILE}, batch_size=32, num_workers=0)
    data_module.setup(stage="test")
    dataloader = data_module.test_dataloader()

    num_batches = len(dataloader)
    assert num_batches == 1

    batch = next(iter(dataloader))
    batch_data, batch_labels = batch
    # Check batch content
    assert batch_data.shape == torch.Size([3, 2686, 1307, 3])
    # Check batch labels
    assert batch_labels is None


def test_creating_yolo_dataloader_factory_func(beam_widths_path):
    """Test creating a YOLO dataloader using factory function with no labels."""
    didson = DIDSON(ARIS_FILE, beam_widths_path)
    frames = didson.load_frames()

    dataloader, dataset = create_yolo_dataloader(ARIS_FILE)
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
    data_module = ARISDataModule(YOLOARISBatchedDataset, {'aris_filepath': ARIS_FILE}, batch_size=32, num_workers=0)
    data_module.setup(stage="test")
    dataloader = data_module.test_dataloader()

    num_batches = len(dataloader)
    assert num_batches == 1

    batch = next(iter(dataloader))
    # Check batch content
    assert batch[0].shape == torch.Size([3, 3, 960, 512])
    # Check batch labels
    assert batch[1] is None or batch[1].numel() == 0

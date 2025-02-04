import pytest

from fisheye.dataloaders import create_aris_dataloader
from fisheye.dataloaders.register import DataloaderRegistry


def test_aris_dataloader():
    fp = '/Users/madison/Downloads/2018-07-09_173000.aris'
    dataloader = DataloaderRegistry.get_dataloader('aris')
    dataset = dataloader(fp)
    print("Dataset size", len(dataset))


def test_aris_dataloader_factory_func():
    fp = '/Users/madison/Downloads/2018-07-09_173000.aris'
    dataloader, _ = create_aris_dataloader(fp)


def test_yolo_dataloader():
    """Successful case: load a dataloader for YOLOv5."""
    fp = '/Users/madison/Downloads/2018-07-09_173000.aris'
    dataloader = DataloaderRegistry.get_dataloader('yolo')
    dataset = dataloader(fp)


def test_unknown_dataloader():
    """Failure case: load an unknown dataloader."""
    model_name = "unknown_model"
    with pytest.raises(ValueError, match="No dataloader found for model: unknown_model"):
        DataloaderRegistry.get_dataloader(model_name)


def test_loading_corrupted_data_to_dataloader():
    pass


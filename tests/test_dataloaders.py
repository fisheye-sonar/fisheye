import pytest
import sys
import os

sys.path.append("/Users/mahobley/Code/fisheye")
from fisheye.dataloaders.aris import create_aris_dataloader
from fisheye.dataloaders.register import DataloaderRegistry


def test_aris_dataloader():
    fp = "/Users/mahobley/Code/salmon_counting_data/RO_2018-05-26_073004.aris"
    beam_width_dir = "/Users/mahobley/Code/salmon_counting_data/beam_widths"
    dataloader = DataloaderRegistry.get_dataloader("aris")
    dataset = dataloader(fp, beam_width_dir=beam_width_dir, return_unwarped=True)
    print("Dataset size", len(dataset))

    for i, (frames, labels) in enumerate(dataset):
        # if labels is not None:
        print(i, frames.shape, labels)


def test_aris_dataloader_factory_func():
    fp = "/Users/mahobley/Code/salmon_counting_data/RO_2018-05-26_073004.aris"
    beam_width_dir = "/Users/mahobley/Code/salmon_counting_data/beam_widths"
    dataloader, _ = create_aris_dataloader(
        fp, beam_width_dir=beam_width_dir, return_unwarped=True
    )

    for i, (frames, labels) in enumerate(dataloader):
        print(i, frames.shape)


def test_yolo_dataloader():
    """Successful case: load a dataloader for YOLOv5."""
    fp = "/Users/mahobley/Code/salmon_counting_data/RO_2018-05-26_073004.aris"
    beam_width_dir = "/Users/mahobley/Code/salmon_counting_data/beam_widths"
    dataloader = DataloaderRegistry.get_dataloader("yolo")
    dataset = dataloader(fp, beam_width_dir=beam_width_dir, return_unwarped=True)
    for i, (batch) in enumerate(dataset):
        ims = [batch[j][0] for j in range(len(batch))]
        labels = [batch[j][1] for j in range(len(batch))]
        shapes = [batch[j][2][1] for j in range(len(batch))]
        print(i, ims[0].shape, labels[0].shape, shapes[0])


def test_unknown_dataloader():
    """Failure case: load an unknown dataloader."""
    model_name = "unknown_model"
    with pytest.raises(
        ValueError, match="No dataloader found for model: unknown_model"
    ):
        DataloaderRegistry.get_dataloader(model_name)


def test_loading_corrupted_data_to_dataloader():
    pass


test_aris_dataloader()
# test_aris_dataloader_factory_func()
# test_yolo_dataloader()

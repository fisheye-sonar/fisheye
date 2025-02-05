import os
from pathlib import Path

import torch

from fisheye.dataloaders.base import BaseDataset
from fisheye.dataloaders.didson.pyDIDSON import DIDSON
from fisheye.dataloaders.samplers import OnePerBatchSampler
from fisheye.utils import torch_distributed_zero_first

BASE = Path(__file__).parent.parent
BEAM_WIDTH_DIR = (BASE / "beam_widths").resolve()


class ARISBatchedDataset(BaseDataset):
    """ARISBatchedDataset

    A Dataset class for loading an ARIS file, loading the frames, and applying background subtraction.
    """

    def __init__(
        self,
        aris_filepath,
        beam_width_dir=BEAM_WIDTH_DIR,
        annotations_file=None,
        batch_size=32,
        num_frames_bg_subtract=1000,
        disable_output=False,
        cache_bg_frames=False,
        do_bg_subtract=True,
    ):
        """
        :param aris_filepath (str): Path to an ARIS file.
        :param beam_width_dir (str): Path to beam widths directory. Defaults to BEAM_WIDTH_DIR.
        :param annotations_file (str): Path to annotations file.
        :param batch_size (int): Batch size. Defaults to 32.
        :param num_frames_bg_subtract: Number of frames to subtract from the background. Defaults to 1000.
        :param disable_output (bool): Whether to disable output. Defaults to False.
        :param cache_bg_frames (bool): Whether to cache background frames. Defaults to False.
        :param do_bg_subtract (bool): Whether to subtract background frames. Defaults to True.
        """

        self.didson = DIDSON(aris_filepath, beam_width_dir=BEAM_WIDTH_DIR)
        start_frame = self.didson.info["startframe"]
        end_frame = self.didson.info["endframe"] or self.didson.info["numframes"]
        xdim, ydim = self.didson.info["xdim"], self.didson.info["ydim"]

        super().__init__(
            start_frame,
            end_frame,
            xdim,
            ydim,
            beam_width_dir,
            annotations_file,
            batch_size,
            num_frames_bg_subtract,
            disable_output,
            cache_bg_frames,
            do_bg_subtract,
        )

    def load_frames(self, start_frame, end_frame):
        """Load ARIS frames."""
        return self.didson.load_frames(start_frame=start_frame, end_frame=end_frame)


def create_aris_dataloader(
    aris_filepath,
    beam_width_dir=BEAM_WIDTH_DIR,
    annotations_file=None,
    batch_size=32,
    rank=-1,
    world_size=1,
    workers=0,
    disable_output=False,
    cache_bg_frames=False,
    do_bg_subtract=True,
):
    """
    Get a PyTorch Dataset and DataLoader for ARIS files with (optional) associated fisheye-formatted labels.
    """
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    # this is a no-op for a single-gpu machine
    with torch_distributed_zero_first(rank):
        dataset = ARISBatchedDataset(
            aris_filepath,
            beam_width_dir,
            annotations_file,
            batch_size=batch_size,
            disable_output=disable_output,
            cache_bg_frames=cache_bg_frames,
            do_bg_subtract=do_bg_subtract,
        )
    batch_size = min(batch_size, len(dataset))
    nw = min(
        [os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers]
    )  # number of workers

    if not disable_output:
        print("Dataset size", len(dataset))
        print("Num workers", nw)

    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset,
        batch_size=None,
        sampler=OnePerBatchSampler(data_source=dataset, batch_size=batch_size),
        num_workers=nw,
        pin_memory=True,
    )
    return dataloader, dataset

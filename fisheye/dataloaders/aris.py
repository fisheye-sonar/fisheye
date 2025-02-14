import os

import torch

from fisheye.dataclasses import ARISDatasetConfig
from fisheye.dataloaders.base import BaseDataset
from fisheye.dataloaders.didson.pyDIDSON import DIDSON
from fisheye.dataloaders.samplers import OnePerBatchSampler
from fisheye.utils import torch_distributed_zero_first


class ARISBatchedDataset(BaseDataset):
    """ARISBatchedDataset

    A Dataset class for loading an ARIS file, loading the frames, and applying background subtraction.
    """

    def __init__(self, config: ARISDatasetConfig):
        """
        :param filepath (str): Path to an ARIS file.
        :param beam_width_dir (str): Path to beam widths directory. Defaults to BEAM_WIDTH_DIR.
        :param annotations_file (str): Path to annotations file.
        :param batch_size (int): Batch size. Defaults to 32.
        :param num_frames_bg_subtract: Number of frames to subtract from the background. Defaults to 1000.
        :param disable_output (bool): Whether to disable output. Defaults to False.
        :param cache_bg_frames (bool): Whether to cache background frames. Defaults to False.
        :param do_bg_subtract (bool): Whether to subtract background frames. Defaults to True.
        :param start_frame (int): Starting frame for ARIS file. Defaults to None.
        :param end_frame (int): Ending frame for ARIS file. Defaults to None.
        """
        try:
            self.didson = DIDSON(config.filepath, beam_width_dir=config.beam_width_dir)
        except Exception as e:
            raise RuntimeError(f"Could not load {config.filepath}: {e}")

        if config.start_frame is None:
            config.start_frame = self.didson.info["startframe"]
        if config.end_frame is None:
            config.end_frame = (
                self.didson.info["endframe"] or self.didson.info["numframes"]
            )

        config.end_frame = min(
            config.end_frame,
            self.didson.info["endframe"] or self.didson.info["numframes"],
        )
        config.xdim, config.ydim = self.didson.info["xdim"], self.didson.info["ydim"]

        super().__init__(config)

    def load_frames(self, start_frame, end_frame, return_unwarped=False):
        """Load ARIS frames."""
        return self.didson.load_frames(
            start_frame=start_frame,
            end_frame=end_frame,
            return_unwarped=return_unwarped,
        )


def create_aris_dataloader(config: ARISDatasetConfig):
    """
    Get a PyTorch Dataset and DataLoader for ARIS files with (optional) associated fisheye-formatted labels.
    """
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    # this is a no-op for a single-gpu machine
    with torch_distributed_zero_first(config.rank):
        dataset = ARISBatchedDataset(config)
    batch_size = min(config.batch_size, len(dataset))
    nw = min(
        [
            os.cpu_count() // config.world_size,
            batch_size if batch_size > 1 else 0,
            config.workers,
        ]
    )  # number of workers

    if not config.disable_output:
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

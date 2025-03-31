import os
import warnings

import torch

from fisheye.configs import ARISDatasetConfig
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

        config.start_frame, config.end_frame = self._validate_frame_range(config=config)

        config.xdim, config.ydim = self.didson.info["xdim"], self.didson.info["ydim"]
        config.image_meter_width = config.xdim * self.didson.info["pixel_meter_width"]
        config.image_meter_height = config.ydim * self.didson.info["pixel_meter_height"]

        super().__init__(config)

    def load_frames(self, start_frame, end_frame, return_unwarped=False):
        """Load ARIS frames."""
        return self.didson.load_frames(
            start_frame=start_frame,
            end_frame=end_frame,
            return_unwarped=return_unwarped,
        )

    def _validate_frame_range(self, config):
        """Validate the start and end frame IDs."""
        end_frame = self.didson.info["numframes"]

        config.end_frame = (
            end_frame
            if not config.end_frame or config.end_frame > end_frame
            else config.end_frame
        )

        if config.end_frame <= 0:
            # If end frame is 0 or -1, something ain't right in the header file. However, there most likely is still
            # data that can be unpacked so load all frames.
            config.end_frame = end_frame
            config.start_frame, config.end_frame = self._validate_frame_range(
                config=config
            )

        # We are possibly looking at a shortened clip where the start and end frame indexes are larger than the number
        # of frames in the file.
        # Also want to check if a user specifies an invalid end_frame
        if config.start_frame > end_frame or config.start_frame > config.end_frame:
            warnings.warn(
                "End frame is 0 or -1, likely due to a corrupted or incomplete header file. "
                "Even if you provided a valid end_frame, it was overwritten because the original end_frame is smaller. "
                "Falling back to loading all frames, which may be inefficient."
            )
            # Reset the start and end frames
            config.start_frame = 0
            config.end_frame = end_frame
            warnings.warn(
                f"Resetting start_frame to 0 and end_frame to {config.end_frame}."
            )

        return config.start_frame, config.end_frame


def create_aris_dataloader(config: ARISDatasetConfig):
    """
    Get a PyTorch Dataset and DataLoader for ARIS files.
    """
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    # this is a no-op for a single-gpu machine
    with torch_distributed_zero_first(config.rank):
        dataset = ARISBatchedDataset(config)

    if len(dataset) == 0:
        warnings.warn(
            "Warning: Dataset contains no valid frames or has incorrect start and end frame indexes, "
            "preventing frame extraction."
        )
        return None, None

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

from pathlib import Path

import numpy as np
import torch

from fisheye.configs import BaseDatasetConfig


class FrameExtractor:
    """Helper for loading frames for a given ARIS file and frame index."""

    def __init__(self, extra_frames: int = 2):
        """
        extra_frames:
            Number of frames after the target frame to request.
            Example: 2 = 1 for background subtraction + 1 because the
            last frame is typically skipped by DIDSON.
        """
        self.extra_frames = extra_frames

    def iter_frames(self, aris_path: Path, frame_idx: int):
        """Yield individual image tensors for the requested frame window."""
        from fisheye.dataloaders import create_dataloader

        config = BaseDatasetConfig(
            filepath=str(aris_path),
            start_frame=frame_idx,
            end_frame=frame_idx + self.extra_frames,
        )
        dataloader, _ = create_dataloader(config)
        for images, *_ in dataloader:
            for image in images:
                yield image


def to_chw_tensor(img):
    """Convert HWC uint8 numpy → CHW tensor."""
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    return torch.from_numpy(img)

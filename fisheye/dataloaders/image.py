import os
import cv2
import numpy as np

from fisheye.dataloaders import BaseDataset


class ImageDataset(BaseDataset):
    """
    A Dataset class for loading images from a folder and applying background subtraction.
    """
    def __init__(self, image_folder, batch_size, num_frames_bg_subtract=1000, do_bg_subtract=True, **kwargs):
        """
        Args:
            image_folder (str): Path to the folder containing images.
            batch_size (int): Number of images in a batch.
            num_frames_bg_subtract (int): Number of frames to use for background subtraction.
            do_bg_subtract (bool): Whether to apply background subtraction or not.
            kwargs: Additional arguments (e.g., annotations file, etc.)
        """
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.do_bg_subtract = do_bg_subtract
        self.num_frames_bg_subtract = num_frames_bg_subtract
        self.image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png'))]
        self.total_frames = len(self.image_paths)

        super().__init__(batch_size=batch_size, num_frames_bg_subtract=num_frames_bg_subtract, do_bg_subtract=do_bg_subtract, **kwargs)

    def __len__(self):
        """Return number of batches in the dataset."""
        return (self.total_frames // self.batch_size) + (1 if self.total_frames % self.batch_size else 0)

    def __getitem__(self, idx):
        """Retrieve a batch of images from the folder."""
        final_idx = min((idx + 1) * self.batch_size, self.total_frames)
        frame_paths = self.image_paths[idx * self.batch_size:final_idx]

        # Load frames from the folder
        frames = [cv2.imread(path) for path in frame_paths]

        if self.do_bg_subtract:
            frames = self._apply_bg_subtraction(np.array(frames))

        return frames

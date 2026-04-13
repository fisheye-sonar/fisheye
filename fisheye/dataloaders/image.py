import dataclasses
import json
import os
import cv2
import numpy as np
import pandas as pd
from fisheye.configs.datasets import ImageDatasetConfig, ARISMetadata
from fisheye.common.generic import run_with_threads
from fisheye.dataloaders.yolo_mixin import YOLOPostprocessMixin


class ImageDataset(YOLOPostprocessMixin):
    """Dataset for loading pre-processed 3-channel images from a folder."""

    def __init__(self, config: ImageDatasetConfig, **kwargs):
        self.image_folder = config.image_folder
        self.return_original_image = config.return_original_image
        self.use_multithreading = config.use_multithreading
        self.max_workers = config.max_workers

        self.image_paths = sorted(
            [
                os.path.join(self.image_folder, filename)
                for filename in os.listdir(self.image_folder)
                if filename.lower().endswith((".jpg", ".jpeg", ".png"))
            ],
            key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]),
        )

        self.total_frames = len(self.image_paths)
        self.start_frame = getattr(config, "start_frame", 0)
        self.end_frame = getattr(config, "end_frame", 0)
        if self.end_frame == 0 or self.end_frame > self.total_frames:
            self.end_frame = self.total_frames

        self.pad = getattr(config, "pad", 0.5)
        self.img_size = getattr(config, "img_size", 896)
        self.stride = getattr(config, "stride", 64)
        self.batch_size = config.batch_size

        self.metadata = self._extract_metadata(config)
        self.original_shape = (
            (self.metadata.ydim, self.metadata.xdim)
            if self.metadata.ydim > 0
            else self._get_shape_from_first_image()
        )
        self.shape = self._compute_resized_shape()

    def _get_shape_from_first_image(self):
        """Read shape from the first image when no metadata JSON is available."""
        if not self.image_paths:
            return (0, 0)
        img = cv2.cvtColor(cv2.imread(self.image_paths[0]), cv2.COLOR_BGR2RGB)
        return img.shape[:2]

    def _extract_metadata(self, config) -> ARISMetadata:
        if getattr(config, "metadata_file", "") and os.path.exists(
            config.metadata_file
        ):
            with open(config.metadata_file, "r") as f:
                info = json.load(f)
            valid_keys = {f.name for f in dataclasses.fields(ARISMetadata)}
            filtered = {k: v for k, v in info.items() if k in valid_keys}
            if "beam_width_data" in filtered and isinstance(
                filtered["beam_width_data"], list
            ):
                filtered["beam_width_data"] = pd.DataFrame(filtered["beam_width_data"])
            return ARISMetadata(**filtered)

        return ARISMetadata(
            xdim=0,
            ydim=0,
            image_meter_width=0,
            image_meter_height=0,
            pixel_meter_size=0,
            x_meter_start=0,
            x_meter_stop=0,
            y_meter_start=0,
            y_meter_stop=0,
            sampleperiod=0,
            soundspeed=0,
            windowstart=0,
            samplesperbeam=0,
            BeamCount=0,
            thesystemtype=0,
            largelens=0,
            numframes=self.total_frames,
            unwarped_shape=(0, 0),
            beam_width_data=None,
        )

    def __len__(self):
        return self.end_frame - self.start_frame

    def __getitem__(self, idx: int):
        final_idx = min(idx + self.batch_size, len(self))
        frames, unwarped_frames = self.load_frames(
            self.start_frame + idx,
            self.start_frame + final_idx,
        )
        return self._postprocess(
            frames, None, unwarped_frames, None, self.return_original_image
        )

    def _load_single_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Bad image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_frames(self, start_idx, end_idx, return_unwarped=False):
        paths = self.image_paths[start_idx:end_idx]

        if not paths:
            return np.empty((0, 0, 0)), np.empty((0, 0, 0))

        if self.use_multithreading and len(paths) > 1:
            frames = run_with_threads(self._load_single_image, paths, self.max_workers)
        else:
            frames = [self._load_single_image(p) for p in paths]

        frames = np.stack(frames)
        return frames, frames  # mimic ARIS structure

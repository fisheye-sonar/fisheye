import dataclasses
import json
import os
import cv2
import numpy as np
import pandas as pd
import torch
from fisheye.configs.datasets import ImageDatasetConfig, ARISMetadata

from fisheye.dataloaders.utils import to_chw_tensor
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import xyxy2xywh


class ImageDataset:
    """
    A Dataset class for loading images from a folder.
    Mirrors BaseDataset.__getitem__ logic without inheriting from it.
    """

    def __init__(self, config: ImageDatasetConfig, **kwargs):
        self.image_folder = config.image_folder
        self.do_bg_subtract = config.do_bg_subtract
        self.num_frames_bg_subtract = config.num_frames_bg_subtract
        self.return_echogram_with_bg_subtracted = (
            config.return_echogram_with_bg_subtracted
        )
        self.return_original_image = config.return_original_image
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

        self.metadata = self._extract_metadata(config)
        self.original_shape = (
            (self.metadata.ydim, self.metadata.xdim)
            if self.metadata.ydim > 0
            else self._get_shape_from_first_image()
        )
        self.shape = self._compute_resized_shape()
        self.batch_size = config.batch_size

    def _get_shape_from_first_image(self):
        """Get the shape of the first image in the dataset if metadata JSON does not exist."""
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
        """Length of the dataset."""
        return self.end_frame - self.start_frame

    def __getitem__(self, idx: int):
        """Retrieve a batch of frames and labels."""
        final_idx = min(idx + self.batch_size, len(self))

        frames, unwarped_frames = self.load_frames(
            self.start_frame + idx,
            self.start_frame + final_idx,
        )

        # Images are already 3-channel — no extra frame or expand_dims needed.
        frame_images = frames
        frame_labels = None
        echogram = None

        postprocessed = self._postprocess(
            frame_images,
            frame_labels,
            unwarped_frames,
            echogram,
            self.return_original_image,
        )
        return postprocessed

    def load_frames(self, start_idx, end_idx, return_unwarped=False):
        paths = self.image_paths[start_idx:end_idx]

        frames = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                raise ValueError(f"Bad image: {p}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        if len(frames) == 0:
            return np.empty((0, 0, 0)), np.empty((0, 0, 0))

        frames = np.stack(frames)

        return frames, frames  # mimic ARIS structure

    @classmethod
    def load_image(cls, img, img_size=896, return_original_image=False):
        """Loads and resizes an image for YOLOv5 inference."""
        img_original = img.copy() if return_original_image else None

        h0, w0 = img.shape[:2]  # original height and width
        r = img_size / max(h0, w0)  # resize ratio
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        resized_img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

        # returns resized_img, original hw, resized hw, original img
        return resized_img, (h0, w0), resized_img.shape[:2], img_original

    def _compute_resized_shape(self):
        """Computes the shape for resizing images based on aspect ratio."""
        aspect_ratio = self.original_shape[0] / self.original_shape[1]
        shape = [1, 1 / aspect_ratio] if aspect_ratio > 1 else [aspect_ratio, 1]
        return (
            np.ceil(np.array(shape) * self.img_size / self.stride + self.pad).astype(
                int
            )
            * self.stride
        )

    def _postprocess(
        self,
        frame_images,
        frame_labels,
        unwarped_frames,
        echogram,
        return_original_image,
    ):
        """
        Return a batch of data in the format used by ScaledYOLOv4.
        That is, a list of tuples, one tuple per image in the batch:
            [
                (resized_img_for_yolo ->torch.Tensor,
                labels ->torch.Tensor,
                shapes ->tuple describing image original dimensions and scaled/padded dimensions
                original_img_tensor ->torch.Tensor (optional)
                ),
                ...
            ]
        """
        outputs = []
        frame_labels = frame_labels or [None for _ in frame_images]
        _debug_first = True
        for image, labels in zip(frame_images, frame_labels):
            resized_img, (h0, w0), (h, w), img_original = self.load_image(
                image, return_original_image=return_original_image
            )
            resized_img, ratio, pad = letterbox(
                resized_img, self.shape, auto=False, scaleup=False, stride=self.stride
            )
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            resized_img_tensor = to_chw_tensor(resized_img)
            original_img_tensor = (
                to_chw_tensor(img_original).float()
                if img_original is not None
                else None
            )

            labels_out = self._process_labels(
                labels, ratio, pad, resized_img_tensor.shape
            )
            outputs.append(
                (resized_img_tensor, labels_out, shapes, original_img_tensor)
            )
        return outputs

    def _process_labels(self, labels, ratio, pad, img_shape):
        """Processes and converts labels from normalized xywh to pixel xyxy format, applies padding from letterbox."""
        if labels is not None and labels.size > 0:
            labels = labels.copy()
            labels[:, 1] = (
                ratio[0] * img_shape[1] * (labels[:, 1] - labels[:, 3] / 2) + pad[0]
            )
            labels[:, 2] = (
                ratio[1] * img_shape[0] * (labels[:, 2] - labels[:, 4] / 2) + pad[1]
            )
            labels[:, 3] = (
                ratio[0] * img_shape[1] * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
            )
            labels[:, 4] = (
                ratio[1] * img_shape[0] * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
            )

            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img_shape[1]
            labels[:, [1, 3]] /= img_shape[2]

            labels_out = torch.zeros((len(labels), 6))
            labels_out[:, 1:] = torch.from_numpy(labels)
            return labels_out

        return torch.zeros((0, 6))

import json

import numpy as np
import cv2
from torch.utils.data import Dataset
from yolov5.utils.general import xyxy2xywh

from fisheye.dataclasses import BaseDatasetConfig
import warnings


class BaseDataset(Dataset):
    """BaseDataset

    Base class for all datasets.
    """

    def __init__(self, config: BaseDatasetConfig):
        """
        :param start_frame (int): Index of the start frame.
        :param end_frame (int): Index of the end frame.
        :param xdim (int): X dimension.
        :param ydim (int): Y dimension.
        :param beam_width_dir (str): Path to beam widths directory. Defaults to BEAM_WIDTH_DIR.
        :param annotations_file (str): Path to annotations file.
        :param batch_size (int): Batch size. Defaults to 32.
        :param num_frames_bg_subtract: Number of frames to subtract from the background image. Defaults to 1000.
        :param disable_output (bool): Whether to disable output. Defaults to False.
        :param cache_bg_frames (bool): Whether to cache background frames. Defaults to False.
        :param do_bg_subtract (bool): Whether to subtract background frames. Defaults to True.
        """

        self.start_frame = config.start_frame
        self.end_frame = config.end_frame
        self.xdim = config.xdim
        self.ydim = config.ydim
        self.beam_width_dir = config.beam_width_dir
        self.batch_size = config.batch_size
        self.disable_output = config.disable_output
        self.cache_bg_frames = config.cache_bg_frames
        self.num_frames_bg_subtract = config.num_frames_bg_subtract
        self.do_bg_subtract = config.do_bg_subtract
        self.extracted_frames = []
        self.frame_labels = []
        self.extracted_unwarped_frames = []
        self.extracted_echograms = []
        self.return_unwarped = config.return_unwarped
        self.return_echogram = config.return_echogram

        if self.return_unwarped and config.annotations_file is not None:
            warnings.warn(
                "Labels from the annotations file will be ignored when return_unwarped is True."
            )

        self._initialize_labels(config.annotations_file)
        self._init_bg_frame()

    def _initialize_labels(self, annotations_file):
        """Load labels from a fisheye-formatted JSON file."""
        if annotations_file is None:
            self.labels = None
        else:
            self._load_labels(annotations_file)

    def _load_labels(self, fisheye_json):
        """Load labels from a fisheye-formatted json file into self.labels in normalized xywh format."""
        js = json.load(open(fisheye_json, "r"))
        labels = []

        for frame in js["frames"]:
            l = []
            for fish in frame["fish"]:
                x, y, w, h = xyxy2xywh(fish["bbox"])
                cx = x + w / 2.0
                cy = y + h / 2.0
                # Each row is `class x_center y_center width height` format. (Normalized)
                l.append([0, cx, cy, w, h])

            l = np.array(l, dtype=np.float32)
            if len(l) == 0:
                l = np.zeros((0, 5), dtype=np.float32)

            labels.append(l)

        if self.return_unwarped:
            # labels are ignored when returning unwarped frames as they are defined in a different axis system
            self.labels = None
        else:
            self.labels = labels
        self.start_frame = js["start_frame"]
        self.end_frame = js["end_frame"]

    def _init_bg_frame(self):
        """Initialize background frame for subtraction."""
        if self.do_bg_subtract or self.return_echogram:
            num_frames_bg = min(
                self.end_frame - self.start_frame,
                self.num_frames_bg_subtract // self.batch_size * self.batch_size + 1,
            )
            frames_for_bg_subtract, unwarped_frames_for_bg_subtract = self.load_frames(
                self.start_frame, self.start_frame + num_frames_bg, return_unwarped=True
            )

            if self.return_unwarped or self.return_echogram:
                (
                    self.unwarped_mean_blurred_frame,
                    self.unwarped_mean_normalization_value,
                ) = self._compute_bg_subtraction(unwarped_frames_for_bg_subtract)

        if self.do_bg_subtract:

            self.mean_blurred_frame, self.mean_normalization_value = (
                self._compute_bg_subtraction(frames_for_bg_subtract)
            )

    def _compute_bg_subtraction(self, frames_for_bg_subtract):
        """Calculate the mean blurred frame and normalization value."""
        mean_blurred_frame = np.zeros(
            [frames_for_bg_subtract.shape[1], frames_for_bg_subtract.shape[2]],
            dtype=np.float32,
        )
        max_blurred_frame = np.zeros(
            [frames_for_bg_subtract.shape[1], frames_for_bg_subtract.shape[2]],
            dtype=np.float32,
        )

        for i in range(frames_for_bg_subtract.shape[0]):
            blurred = cv2.GaussianBlur(frames_for_bg_subtract[i], (5, 5), 0)
            mean_blurred_frame += blurred
            max_blurred_frame = np.maximum(max_blurred_frame, np.abs(blurred))

        mean_blurred_frame /= frames_for_bg_subtract.shape[0]
        max_blurred_frame -= mean_blurred_frame
        mean_normalization_value = np.max(max_blurred_frame)

        return mean_blurred_frame, mean_normalization_value

    def __len__(self):
        """Length of the dataset excluding the last frame."""
        return self.end_frame - self.start_frame - 1

    def __getitem__(self, idx: int):
        """Retrieve a batch of frames and labels."""
        final_idx = min(idx + self.batch_size, len(self))
        frame_labels = self.labels[idx:final_idx] if self.labels else None

        if idx + 1 < len(self.extracted_frames):
            return self._postprocess(
                np.stack(self.extracted_frames[idx:final_idx]),
                frame_labels,
                np.stack(self.extracted_unwarped_frames[idx:final_idx]),
                np.stack(self.extracted_echograms[idx:final_idx]),
            )

        else:

            frames, unwarped_frames = self.load_frames(
                self.start_frame + idx,
                self.start_frame + final_idx + 1,
                return_unwarped=self.return_unwarped or self.return_echogram,
            )

            if self.return_unwarped:
                frame_images = unwarped_frames
            else:
                frame_images = frames

            # MAH 2025-02-07 17:13:36 Question, why are we removing the last frame?
            # whether or not we are doing background subtraction the image is 4D (previous behaviour was 4D for background subtracted was [t,h,w, 3] not was [t,h,w])
            frame_images = (
                self._apply_bg_subtraction(frame_images)
                if self.do_bg_subtract
                else np.expand_dims(frame_images[:-1], -1)
            )
            if self.return_unwarped or self.return_echogram:
                unwarped_frames = unwarped_frames[:-1]

            if self.return_echogram:
                echogram = self._get_echogram(unwarped_frames)
            else:
                echogram = None

            if self.cache_bg_frames:
                self.extracted_frames.extend(frame_images)
                if frame_labels is not None:
                    self.frame_labels.extend(frame_labels)
                else:
                    self.frame_labels = None
                    self.extracted_unwarped_frames.extend(unwarped_frames)
                self.extracted_echograms.extend(echogram)

        # MAH 2025-02-07 16:48:40 I think this is likely the best solution, it means indexes will be consistent and if needed we can add more things to the list when required
        return self._postprocess(frame_images, frame_labels, unwarped_frames, echogram)

    def _apply_bg_subtraction(self, frames: np.ndarray):
        """Apply background subtraction."""
        # MAH 2025-02-05 19:16:34 TODO this function should be renamed to something that describes the fact it is stacking the channels
        blurred_frames = frames.astype(np.float32)
        for i in range(frames.shape[0]):
            blurred_frames[i] = cv2.GaussianBlur(blurred_frames[i], (5, 5), 0)
        if self.return_unwarped:
            blurred_frames -= self.unwarped_mean_blurred_frame
            blurred_frames /= self.unwarped_mean_normalization_value
        else:
            blurred_frames -= self.mean_blurred_frame
            blurred_frames /= self.mean_normalization_value
        blurred_frames += 1
        blurred_frames /= 2

        frame_image = np.stack(
            [
                frames[:-1],
                blurred_frames[:-1] * 255,
                np.abs(blurred_frames[1:] - blurred_frames[:-1]) * 255,
            ],
            axis=-1,
        ).astype(np.uint8, copy=False)

        return frame_image

    def _postprocess(self, frame_images, frame_labels, unwarped_frames, echogram):
        """Postprocess frames before returning."""
        return frame_images, frame_labels, unwarped_frames, echogram

    def _get_echogram(self, unwarped_frames):
        """Generate Echogram from the frames.
        the return channels are the magnitude and the normalised angle between -0.5 and 0.5
        """
        unwarped_frames_bgs = unwarped_frames.astype(np.float32)
        unwarped_frames_bgs -= self.unwarped_mean_blurred_frame
        unwarped_frames_bgs /= self.unwarped_mean_normalization_value
        echogram = np.max(unwarped_frames_bgs.astype(np.float32), axis=2).astype(
            np.float32
        )
        col = np.argmax(unwarped_frames_bgs, axis=2).astype(np.float32)
        col = col / unwarped_frames.shape[2]
        col -= 0.5
        echogram = np.stack([echogram, col], axis=2)

        return echogram

    def load_frames(self, idx, final_idx, return_unwarped):
        raise NotImplementedError("Subclasses should implement this method.")

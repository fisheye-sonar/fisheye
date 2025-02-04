import json
from pathlib import Path

import numpy as np
import cv2
from torch.utils.data import Dataset
from fisheye.lib.yolo import xyxy2xywh

BASE = Path(__file__).parent.parent
BEAM_WIDTH_DIR = (BASE / "beam_widths").resolve()


class BaseDataset(Dataset):
    def __init__(self, start_frame, end_frame, xdim, ydim, beam_width_dir=BEAM_WIDTH_DIR, annotations_file=None, batch_size=32,
                 num_frames_bg_subtract=1000,
                 disable_output=False, cache_bg_frames=False, do_bg_subtract=True):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.xdim = xdim
        self.ydim = ydim
        self.beam_width_dir = beam_width_dir
        self.batch_size = batch_size
        self.disable_output = disable_output
        self.cache_bg_frames = cache_bg_frames
        self.num_frames_bg_subtract = num_frames_bg_subtract
        self.do_bg_subtract = do_bg_subtract
        self.extracted_frames = []

        self._initialize_labels(annotations_file)
        self._init_bg_frame()

    def _initialize_labels(self, annotations_file):
        """Load labels from a fisheye-formatted JSON file."""
        if annotations_file is None:
            self.labels = None
        else:
            self._load_labels(annotations_file)

    def _load_labels(self, fisheye_json):
        """Load labels from a fisheye-formatted json file into self.labels in normalized xywh format."""
        js = json.load(open(fisheye_json, 'r'))
        labels = []

        for frame in js['frames']:
            l = []
            for fish in frame['fish']:
                x, y, w, h = xyxy2xywh(fish['bbox'])
                cx = x + w/2.0
                cy = y + h/2.0
                # Each row is `class x_center y_center width height` format. (Normalized)
                l.append([0, cx, cy, w, h])

            l = np.array(l, dtype=np.float32)
            if len(l) == 0:
                l = np.zeros((0, 5), dtype=np.float32)

            labels.append(l)

        self.labels = labels
        self.start_frame = js['start_frame']
        self.end_frame = js['end_frame']

    def _init_bg_frame(self):
        """Initialize background frame for subtraction."""
        if not self.do_bg_subtract:
            return

        num_frames_bg = min(self.end_frame - self.start_frame,
                            self.num_frames_bg_subtract // self.batch_size * self.batch_size + 1)
        frames_for_bg_subtract = self.load_frames(self.start_frame, self.start_frame + num_frames_bg)

        self.mean_blurred_frame, self.mean_normalization_value = self._compute_bg_subtraction(frames_for_bg_subtract)

    def _compute_bg_subtraction(self, frames_for_bg_subtract):
        """Calculate the mean blurred frame and normalization value."""
        mean_blurred_frame = np.zeros([self.ydim, self.xdim], dtype=np.float32)
        max_blurred_frame = np.zeros([self.ydim, self.xdim], dtype=np.float32)

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
            return self._postprocess(self.extracted_frames[idx:final_idx], frame_labels)
        else:
            frames = self.load_frames(self.start_frame+idx, self.start_frame + final_idx + 1)
            frame_images = self._apply_bg_subtraction(frames) if self.do_bg_subtract else frames[:-1]

            if self.cache_bg_frames:
                self.extracted_frames.extend(frame_images)

        return self._postprocess(frame_images, frame_labels)

    def _apply_bg_subtraction(self, frames: np.ndarray):
        """Apply background subtraction."""
        blurred_frames = frames.astype(np.float32)
        for i in range(frames.shape[0]):
            blurred_frames[i] = cv2.GaussianBlur(blurred_frames[i], (5, 5), 0)
        blurred_frames -= self.mean_blurred_frame
        blurred_frames /= self.mean_normalization_value
        blurred_frames += 1
        blurred_frames /= 2

        frame_image = np.stack(
            [frames[:-1], blurred_frames[:-1] * 255, np.abs(blurred_frames[1:] - blurred_frames[:-1]) * 255],
            axis=-1).astype(np.uint8, copy=False)

        return frame_image

    def _postprocess(self, frame_images, frame_labels):
        """Postprocess frames before returning."""
        return frame_images, frame_labels

    def load_frames(self, idx, final_idx):
        raise NotImplementedError("Subclasses should implement this method.")

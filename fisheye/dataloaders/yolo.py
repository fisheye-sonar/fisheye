from pathlib import Path

import numpy as np
import torch
import cv2

from fisheye.dataloaders import ARISBatchedDataset
from fisheye.lib.yolo import xyxy2xywh, letterbox


BASE = Path(__file__).parent.parent
BEAM_WIDTH_DIR = (BASE / "beam_widths").resolve()


class YOLOARISBatchedDataset(ARISBatchedDataset):
    """YOLOARISBatchedDataset

    An ARIS Dataset tailored for YOLOv5 inference."""

    def __init__(
        self,
        aris_filepath,
        beam_width_dir=BEAM_WIDTH_DIR,
        annotations_file=None,
        stride=64,
        pad=0.5,
        img_size=896,
        batch_size=32,
        disable_output=False,
        cache_bg_frames=False,
    ):
        """
        :param aris_filepath (str): Path to an ARIS file.
        :param beam_width_dir (str): Path to beam widths directory. Defaults to BEAM_WIDTH_DIR.
        :param annotations_file (str): Path to annotations file.
        :param stride (int): Stride size for YOLOv5 inference. Defaults to 64.
        :param pad (float): Pad size for YOLOv5 inference. Defaults to 0.5.
        :param img_size (int): Image size for YOLOv5 inference. Defaults to 896.
        :param batch_size (int): Batch size. Defaults to 32.
        :param disable_output (bool): Whether to disable output. Defaults to False.
        :param cache_bg_frames (bool): Whether to cache background frames. Defaults to False.
        """
        super().__init__(
            aris_filepath,
            beam_width_dir,
            annotations_file,
            batch_size,
            disable_output=disable_output,
            cache_bg_frames=cache_bg_frames,
        )

        self.stride = stride
        self.pad = pad
        self.img_size = img_size
        self.original_shape = (self.ydim, self.xdim)
        self.shape = self._compute_resized_shape()

    def _compute_resized_shape(self):
        """Computes the shape for resizing images based on aspect ratio."""
        aspect_ratio = self.ydim / self.xdim
        shape = [1, 1 / aspect_ratio] if aspect_ratio > 1 else [aspect_ratio, 1]
        return (
            np.ceil(np.array(shape) * self.img_size / self.stride + self.pad).astype(
                int
            )
            * self.stride
        )

    @classmethod
    def load_image(cls, img, img_size=896):
        """Loads and resizes an image for YOLOv5 inference."""
        h0, w0 = img.shape[:2]  # original height and width
        r = img_size / max(h0, w0)  # resize ratio
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # returns img, original hw, resized hw

    def _postprocess(self, frame_images, frame_labels):
        """
        Return a batch of data in the format used by ScaledYOLOv4.
        That is, a list of tuples, on tuple per image in the batch:
            [
                (img ->torch.Tensor,
                labels ->torch.Tensor,
                shapes ->tuple describing image original dimensions and scaled/padded dimensions
                ),
                ...
            ]
        """
        outputs = []
        frame_labels = frame_labels or [None for _ in frame_images]

        for image, labels in zip(frame_images, frame_labels):
            img, (h0, w0), (h, w) = self.load_image(image)

            # Apply letterboxing to resize and pad images
            img, ratio, pad = letterbox(img, self.shape, auto=False, scaleup=False)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            img = img.transpose(2, 0, 1)  # Convert to CxHxW format
            img = np.ascontiguousarray(img)

            labels_out = self._process_labels(labels, ratio, pad, img.shape)
            outputs.append((torch.from_numpy(img), labels_out, shapes))

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

            # Convert to xywh format and normalize
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img_shape[1]  # Normalize height
            labels[:, [1, 3]] /= img_shape[2]  # Normalize width

            labels_out = torch.zeros((len(labels), 6))
            labels_out[:, 1:] = torch.from_numpy(labels)
            return labels_out

        return torch.zeros((0, 6))

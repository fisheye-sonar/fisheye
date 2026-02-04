import cv2
import numpy as np
import structlog
import torch
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import xyxy2xywh

from fisheye.configs import YOLODatasetConfig
from fisheye.dataloaders import ARISBatchedDataset
from fisheye.dataloaders.utils import to_chw_tensor
from fisheye.dataloaders.didson.pyDIDSON import compute_resized_shape

logger = structlog.get_logger()


class CenterPad:
    def __init__(self, in_hw, out_hw, pad_value=114):
        self.in_h, self.in_w = in_hw
        self.out_h, self.out_w = out_hw
        self.pad_value = pad_value

        if self.in_h > self.out_h or self.in_w > self.out_w:
            raise ValueError(
                f"Input ({self.in_h},{self.in_w}) larger than target ({self.out_h},{self.out_w}); pads only."
            )

        self.dh = (self.out_h - self.in_h) // 2
        self.dw = (self.out_w - self.in_w) // 2
        self.top = self.dh
        self.left = self.dw

    def __call__(self, img: np.ndarray):
        out = np.full(
            (
                (self.out_h, self.out_w, img.shape[2])
                if img.ndim == 3
                else (self.out_h, self.out_w)
            ),
            self.pad_value,
            dtype=img.dtype,
        )
        # out[...] = self.pad_value
        out[self.top : self.top + self.in_h, self.left : self.left + self.in_w, ...] = (
            img
        )
        return out, (1.0, 1.0), (float(self.dw), float(self.dh))


class YOLOARISBatchedDataset(ARISBatchedDataset):
    """YOLOARISBatchedDataset

    A PyTorch Dataset for loading ARIS/DIDSON data specifically tailored for YOLOv5-style object detection tasks.
    """

    def __init__(self, config: YOLODatasetConfig):
        """
        Initialize the YOLOARISBatchedDataset with YOLO-specific configuration options.

        Args:
            config (YOLODatasetConfig): Configuration object containing all dataset parameters optimized.
        """
        super().__init__(config)

        self.stride = config.stride
        self.pad = config.pad
        self.model_input_img_size = config.img_size
        self.original_shape = (self.metadata.ydim, self.metadata.xdim)
        print(f"{self.metadata.ydim=} {self.metadata.xdim=}")
        self.resize_image_shape = compute_resized_shape(
            self.original_shape, self.model_input_img_size, self.stride
        )
        print(f"{self.model_input_img_size=} {self.resize_image_shape=}")

        self.pad = CenterPad(self.resize_image_shape, self.model_input_img_size)

    def _postprocess(
        self,
        frame_images,
        frame_labels,
        unwarped_frames,
        echogram,
        original_frames,
    ):
        """
        Return a batch of data in the format used by ScaledYOLOv4.
        That is, a list of tuples, on tuple per image in the batch:
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
        if original_frames is None:
            original_frames = [None for _ in frame_images]
        for image, labels, original_frame in zip(
            frame_images, frame_labels, original_frames
        ):
            # resized_img, original hw, resized hw, original img
            resized_img = image
            # MAH 2026-02-04 10:13:24 TODO should one of these be the original frame size?
            h0, w0 = image.shape[:2]
            h, w = image.shape[:2]
            img_original = original_frame

            # Apply just padding instead of letterboxing
            resized_img, ratio, pad = self.pad(resized_img)

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

            # Convert to xywh format and normalize
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img_shape[1]  # Normalize height
            labels[:, [1, 3]] /= img_shape[2]  # Normalize width

            labels_out = torch.zeros((len(labels), 6))
            labels_out[:, 1:] = torch.from_numpy(labels)
            return labels_out

        return torch.zeros((0, 6))

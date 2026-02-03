import cv2
import numpy as np
import structlog
import torch
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import xyxy2xywh

from fisheye.configs import YOLODatasetConfig
from fisheye.dataloaders import ARISBatchedDataset
from fisheye.dataloaders.utils import to_chw_tensor

logger = structlog.get_logger()


def pad_to_shape(img: np.ndarray, out_hw, pad_value=114):
    """
    Center-pad HWC image to (out_h, out_w). No resizing.
    Returns: padded_img, (1.0, 1.0), (dw, dh) where dw/dh are half-pads.
    """
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    h, w = img.shape[:2]

    if h > out_h or w > out_w:
        raise ValueError(
            f"Input ({h},{w}) larger than target ({out_h},{out_w}); this function pads only."
        )

    dh = (out_h - h) // 2
    dw = (out_w - w) // 2
    top, bottom = dh, out_h - h - dh
    left, right = dw, out_w - w - dw

    # Create output and fill
    if img.ndim == 3:
        out = np.empty((out_h, out_w, img.shape[2]), dtype=img.dtype)
    else:
        out = np.empty((out_h, out_w), dtype=img.dtype)

    # Fill background
    # If pad_value is scalar, numpy broadcasts; if tuple/list matches channels, also broadcasts.
    out[...] = pad_value

    # Paste original image
    out[top : top + h, left : left + w, ...] = img

    return out, (1.0, 1.0), (float(dw), float(dh))


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
        self.resize_image_shape = self._compute_resized_shape()
        print(f"{self.resize_image_shape=}")

    def _compute_resized_shape(self, snap_to_stride: bool = False):
        """
        Compute resized (H, W) that preserves aspect ratio and fits inside img_size.
        One dimension will equal img_size, the other will be <= img_size.
        (You will pad to img_size afterward.)

        If snap_to_stride=True, the non-saturating dimension is floored to a stride multiple
        (and the saturating dimension is kept at img_size).
        """
        oh, ow = int(self.original_shape[0]), int(self.original_shape[1])
        out_h, out_w = int(self.model_input_img_size[0]), int(
            self.model_input_img_size[1]
        )
        stride = int(self.stride)

        # scale to fit within out_h x out_w (keeps aspect ratio)
        s = min(out_h / oh, out_w / ow)

        new_h = int(round(oh * s))
        new_w = int(round(ow * s))

        # guarantee we don't exceed target due to rounding
        new_h = min(new_h, out_h)
        new_w = min(new_w, out_w)

        # Ensure at least 1 pixel
        new_h = max(new_h, 1)
        new_w = max(new_w, 1)

        # Optional: make the "smaller" side a stride multiple (common in some pipelines)
        if snap_to_stride:
            # Only adjust the dimension that is NOT already at its max.
            if new_h < out_h:
                new_h = max(
                    (new_h // stride) * stride, stride if out_h >= stride else 1
                )
            if new_w < out_w:
                new_w = max(
                    (new_w // stride) * stride, stride if out_w >= stride else 1
                )

            # Don’t let snapping push anything over the target
            new_h = min(new_h, out_h)
            new_w = min(new_w, out_w)

        return np.array([new_h, new_w], dtype=int)

    @classmethod
    def load_image(cls, img, img_size=896, return_original_image=False):
        """Loads and resizes an image for YOLOv5 inference."""
        img_original = img.copy() if return_original_image else None
        h0, w0 = img.shape[:2]  # original height and width
        # r = img_size / max(h0, w0)  # resize ratio
        # interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        # resized_img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

        # returns resized_img, original hw, resized hw, original img
        return img_original, (h0, w0), img_original.shape[:2], img_original

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
        for image, labels in zip(frame_images, frame_labels):
            # resized_img, original hw, resized hw, original img
            resized_img = image
            # print(f"{image.shape=}")
            h0, w0 = image.shape[:2]
            h, w = image.shape[:2]
            img_original = None

            if True:
                # Apply just padding instead of letterboxing
                resized_img, ratio, pad = pad_to_shape(
                    resized_img, self.model_input_img_size
                )
            else:
                # Apply letterboxing to resize and pad images
                resized_img, ratio, pad = letterbox(
                    resized_img,
                    self.resize_image_shape,
                    auto=False,
                    scaleup=False,
                    stride=self.stride,
                )
            # print(f"after {resized_img.shape=} {ratio=} {pad=}")

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

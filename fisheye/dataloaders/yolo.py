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
        self.img_size = config.img_size
        self.original_shape = (self.metadata.ydim, self.metadata.xdim)
        self.shape = self._compute_resized_shape()

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
            resized_img, (h0, w0), (h, w), img_original = self.load_image(
                image, return_original_image=return_original_image
            )
            # Apply letterboxing to resize and pad images
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

            # Convert to xywh format and normalize
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img_shape[1]  # Normalize height
            labels[:, [1, 3]] /= img_shape[2]  # Normalize width

            labels_out = torch.zeros((len(labels), 6))
            labels_out[:, 1:] = torch.from_numpy(labels)
            return labels_out

        return torch.zeros((0, 6))

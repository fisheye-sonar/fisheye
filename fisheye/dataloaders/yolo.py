import cv2
import numpy as np
import structlog
import torch
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import xyxy2xywh

from fisheye.configs import YOLODatasetConfig
from fisheye.dataloaders import ARISBatchedDataset

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
    def load_image(cls, img, img_size=896, return_original_shape=False):
        """Loads and resizes an image for YOLOv5 inference."""
        if return_original_shape:
            img_original = img.copy()
        else:
            img_original = None
        h0, w0 = img.shape[:2]  # original height and width
        r = img_size / max(h0, w0)  # resize ratio
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

        return (
            img,
            (h0, w0),
            img.shape[:2],
            img_original,
        )  # returns img, original hw, resized hw, original img

    def _postprocess(self, frame_images, frame_labels, unwarped_frames, echogram):
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
            # MAH 2025-11-25 19:21:01 return the original image shape for the postprocessing step
            img, (h0, w0), (h, w), img_original = self.load_image(
                image, return_original_shape=True
            )

            # Apply letterboxing to resize and pad images
            img, ratio, pad = letterbox(
                img, self.shape, auto=False, scaleup=False, stride=self.stride
            )
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            img = img.transpose(2, 0, 1)  # Convert to CxHxW format
            img = np.ascontiguousarray(img)

            if img_original is not None:
                # MAH 2025-11-25 19:21:01 return the original image, used for length estimation
                img_original = img_original.transpose(
                    2, 0, 1
                )  # Convert to CxHxW format
                img_original = np.ascontiguousarray(img_original)
                img_original = torch.from_numpy(img_original).float()

            labels_out = self._process_labels(labels, ratio, pad, img.shape)
            outputs.append((torch.from_numpy(img), labels_out, shapes, img_original))
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

import cv2
import numpy as np
import torch
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import xyxy2xywh

from fisheye.dataloaders.utils import to_chw_tensor


class YOLOPostprocessMixin:
    """Shared YOLO postprocessing logic for datasets feeding the YOLOv5 detection pipeline.

    Concrete classes must set these instance attributes in __init__:
        shape (np.ndarray): Target image shape after resize + letterbox.
        stride (int): Model stride for letterbox alignment.
        img_size (int): Target size for the longest image edge.
        pad (float): Letterbox padding ratio.
        original_shape (tuple[int, int]): (height, width) of the source images.
    """

    @classmethod
    def load_image(cls, img, img_size=896, return_original_image=False):
        """Resize an image for YOLOv5 inference, preserving aspect ratio."""
        img_original = img.copy() if return_original_image else None

        h0, w0 = img.shape[:2]
        r = img_size / max(h0, w0)
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        resized_img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

        return resized_img, (h0, w0), resized_img.shape[:2], img_original

    def _compute_resized_shape(self):
        """Compute letterbox target shape from the source aspect ratio."""
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
        """Return a batch in YOLOv5 format: list of (img_tensor, labels, shapes, original_img)."""
        outputs = []
        frame_labels = frame_labels or [None for _ in frame_images]
        for image, labels in zip(frame_images, frame_labels):
            resized_img, (h0, w0), (h, w), img_original = self.load_image(
                image, return_original_image=return_original_image
            )
            resized_img, ratio, pad = letterbox(
                resized_img, self.shape, auto=False, scaleup=False, stride=self.stride
            )
            shapes = (h0, w0), ((h / h0, w / w0), pad)

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
        """Convert labels from normalised xywh to pixel xyxy, applying letterbox padding."""
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

import cv2
import numpy as np
import structlog
import math
import torch
import torch.nn.functional as F
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
        self._postprocess = (
            self._postprocess_batchwise
            if config.preprocess_batchwise
            else self._postprocess_serial
        )

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

    def letterbox_batch_torch(self, imgs, new_shape=(640, 640), stride=32):
        """
        imgs: torch.Tensor (N, C, H, W), float32 in [0,1] or [0,255]
        new_shape: (h, w)
        returns:
            imgs_out: (N, C, new_shape[0], new_shape[1])
            ratio: (rw, rh)  # same for all images
            pad: (dw, dh)    # total padding (before division by 2)
        """
        n, c, h, w = imgs.shape

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # scale ratio (new / old), keep aspect ratio
        r = min(new_shape[0] / h, new_shape[1] / w)

        new_unpad_w = int(round(w * r))
        new_unpad_h = int(round(h * r))
        # check if the interpolate is happening on the gpu

        # resize whole batch in one go
        imgs = F.interpolate(
            imgs,
            size=(new_unpad_h, new_unpad_w),
            mode="bilinear",
            align_corners=False,
        )

        # compute padding to reach new_shape
        dw = new_shape[1] - new_unpad_w  # width padding total
        dh = new_shape[0] - new_unpad_h  # height padding total

        # divide padding equally left/right, top/bottom
        left = int(math.floor(dw / 2))
        right = int(math.ceil(dw / 2))
        top = int(math.floor(dh / 2))
        bottom = int(math.ceil(dh / 2))

        # pad order: (left, right, top, bottom)
        imgs = F.pad(imgs, (left, right, top, bottom), value=0.0)

        ratio = (r, r)
        pad = (dw / 2.0, dh / 2.0)  # matches YOLO letterbox convention
        return imgs, ratio, pad

    def _postprocess_batchwise(
        self, frame_images, frame_labels, unwarped_frames, echogram
    ):
        """
        Batched version assuming:
        - all frame_images have same H, W, C
        - all outputs go to the same self.shape
        Returns list of:
            (img: torch.Tensor (C,H',W'),
            labels: torch.Tensor or None,
            shapes: ((h0, w0), ((h'/h0, w'/w0), pad)),
            img_original: torch.Tensor (C, h0, w0))
        """
        # device = getattr(self, "device", "cuda:0")

        # default labels if None
        frame_labels = frame_labels or [None for _ in frame_images]

        # 1) Stack images into a batch: (N, H, W, C) -> (N, C, H, W)
        assert len(frame_images) > 0, "No images provided"

        h0, w0 = frame_images[0].shape[:2]
        # keep originals as torch for later use
        device = torch.device("cuda:0")
        imgs_original = (
            torch.from_numpy(frame_images.transpose(0, 3, 1, 2)).to(device).float()
        )

        # 2) Normalize and move to device

        # 3) Batched letterbox to self.shape
        imgs, ratio, pad = self.letterbox_batch_torch(
            imgs_original,
            new_shape=self.shape,  # e.g. (896, 896)
            stride=self.stride,
        )

        # after letterbox
        _, _, h, w = imgs.shape

        # 4) Build shapes (same for all images if h0, w0 are same)
        shapes = [((h0, w0), ((h / h0, w / w0), pad)) for _ in range(len(frame_images))]

        # 5) Per-image label processing & packaging
        output_labels = []
        for i, labels in enumerate(frame_labels):
            # _process_labels still works per image
            labels_out = self._process_labels(
                labels,
                ratio,
                pad,
                imgs[i].shape,  # (C, H', W')
            )

            output_labels.append(labels_out)

        for i, label in enumerate(output_labels):
            label[:, 0] = i  # add target image index for build_targets()
        output_labels = torch.cat(output_labels, 0)
        imgs = imgs / 255.0
        return (imgs, output_labels, shapes)

    @classmethod
    def load_image(cls, img, img_size=896):
        """Loads and resizes an image for YOLOv5 inference."""
        h0, w0 = img.shape[:2]  # original height and width
        r = img_size / max(h0, w0)  # resize ratio
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # returns img, original hw, resized hw

    def _postprocess_serial(
        self, frame_images, frame_labels, unwarped_frames, echogram
    ):
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
            img, ratio, pad = letterbox(
                img, self.shape, auto=False, scaleup=False, stride=self.stride
            )
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            img = img.transpose(2, 0, 1)  # Convert to CxHxW format
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).float()
            img = img / 255.0

            labels_out = self._process_labels(labels, ratio, pad, img.shape)
            outputs.append(
                (
                    img,
                    labels_out,
                    shapes,
                )
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

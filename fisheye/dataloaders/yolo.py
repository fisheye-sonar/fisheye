import os

import cv2
import numpy as np
import torch
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import xyxy2xywh

from fisheye.configs import YOLODatasetConfig
from fisheye.dataloaders import ARISBatchedDataset
from fisheye.dataloaders.samplers import OnePerBatchSampler
from fisheye.utils import torch_distributed_zero_first, yolo_collate_fn


class YOLOARISBatchedDataset(ARISBatchedDataset):
    """YOLOARISBatchedDataset

    An ARIS Dataset tailored for YOLOv5 inference."""

    def __init__(self, config: YOLODatasetConfig):
        """
        :param filepath (str): Path to an ARIS file.
        :param beam_width_dir (str): Path to beam widths directory. Defaults to BEAM_WIDTH_DIR.
        :param annotations_file (str): Path to annotations file.
        :param stride (int): Stride size for YOLOv5 inference. Defaults to 64.
        :param pad (float): Pad size for YOLOv5 inference. Defaults to 0.5.
        :param img_size (int): Image size for YOLOv5 inference. Defaults to 896.
        :param batch_size (int): Batch size. Defaults to 32.
        :param disable_output (bool): Whether to disable output. Defaults to False.
        :param cache_bg_frames (bool): Whether to cache background frames. Defaults to False.
        :param start_frame (int): Starting frame for ARIS file. Defaults to None.
        :param end_frame (int): Ending frame for ARIS file. Defaults to None.
        """
        super().__init__(config)

        self.stride = config.stride
        self.pad = config.pad
        self.img_size = config.img_size
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
            img, (h0, w0), (h, w) = self.load_image(image)

            # Apply letterboxing to resize and pad images
            img, ratio, pad = letterbox(
                img, self.shape, auto=False, scaleup=False, stride=self.stride
            )
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            img = img.transpose(2, 0, 1)  # Convert to CxHxW format
            img = np.ascontiguousarray(img)

            labels_out = self._process_labels(labels, ratio, pad, img.shape)
            outputs.append(
                (
                    torch.from_numpy(img),
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


def create_yolo_dataloader(config: YOLODatasetConfig):
    """
    Get a PyTorch Dataset and DataLoader for ARIS files with (optional) associated fisheye-formatted labels.
    """
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    # this is a no-op for a single-gpu machine
    with torch_distributed_zero_first(config.rank):
        dataset = YOLOARISBatchedDataset(config)

    batch_size = min(config.batch_size, len(dataset))
    nw = min(
        [
            os.cpu_count() // config.world_size,
            batch_size if batch_size > 1 else 0,
            config.workers,
        ]
    )  # number of workers

    if not config.disable_output:
        print("Dataset size", len(dataset))
        print("Dataset shape", dataset.shape)
        print("Num workers", nw)

    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset,
        batch_size=None,
        sampler=OnePerBatchSampler(data_source=dataset, batch_size=batch_size),
        num_workers=nw,
        pin_memory=True,
        collate_fn=yolo_collate_fn,
    )
    return dataloader, dataset

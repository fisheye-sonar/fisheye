from functools import partial
from typing import Union, List, Tuple, Dict

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from yolov5.utils.general import xywh2xyxy, clip_boxes, scale_boxes
from yolov5.utils.metrics import box_iou

from fisheye.configs.inference import NMSConfig


def convert_xyxy_to_cxcywh(bbox, image_width, image_height):
    """Convert [x1, y1, x2, y2] into YOLO format [x_center, y_center, w, h].

    All returned values are normalized to the range [0, 1].
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2
    y_center = y1 + h / 2

    # Normalize
    x_center /= image_width
    y_center /= image_height
    w /= image_width
    h /= image_height

    return [x_center, y_center, w, h]


def get_bbox_widths_from_yolo_to_image(
    bboxes: Union[torch.Tensor, np.ndarray, List], image_width: int
) -> torch.Tensor:
    """
    Get the bounding box widths in image space from YOLO pixel space.

    Args:
        bboxes: Bounding boxes in YOLO format [x_center, y_center, width, height], normalized [0,1].
        image_width: Width of the image in pixels.

    Returns:
        torch.Tensor: Widths in pixels.
    """
    if bboxes is None or (
        isinstance(bboxes, (list, np.ndarray, torch.Tensor)) and len(bboxes) == 0
    ):
        return torch.tensor([])

    # Convert to torch.Tensor
    if isinstance(bboxes, list):
        bboxes = torch.from_numpy(np.array(bboxes, dtype=np.float32))
    elif isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes).float()
    elif not isinstance(bboxes, torch.Tensor):
        raise TypeError(f"Unsupported type {type(bboxes)}")

    # Handle single bbox (1D)
    if bboxes.ndim == 1:
        bboxes = bboxes.unsqueeze(0)

    # Scale YOLO width to pixels
    return bboxes[:, 2] * image_width


def std_bbox_width_yolo_to_image(
    bboxes: Union[torch.Tensor, np.ndarray, List], image_meter_width: int
) -> float:
    """Calculate the standard deviation bounding box width in image space from YOLO pixel space."""
    bbox_width = get_bbox_widths_from_yolo_to_image(bboxes, image_meter_width)

    return round(bbox_width.std().item(), 2)


def median_bbox_width_yolo_to_image(
    bboxes: Union[torch.Tensor, np.ndarray, List], image_meter_width: int
) -> float:
    """Calculate the median bounding box width in image space from YOLO pixel space."""
    bbox_width = get_bbox_widths_from_yolo_to_image(bboxes, image_meter_width)

    return round(bbox_width.median().item(), 2)


def mean_bbox_width_yolo_to_image(
    bboxes: Union[torch.Tensor, np.ndarray, List], image_meter_width: int
) -> float:
    """Calculate the mean bounding box width in image space from YOLO pixel space."""
    bbox_width = get_bbox_widths_from_yolo_to_image(bboxes, image_meter_width)

    return round(bbox_width.mean().item(), 2)


def norm(bbox, w, h):
    """
    Normalize a bounding box.
    Args:
        bbox: list of length 4. Can be [x,y,w,h] or [x0,y0,x1,y1]
        w: image width
        h: image height
    """
    bb = bbox.copy()
    bb[0] /= w
    bb[1] /= h
    bb[2] /= w
    bb[3] /= h

    return bb


def normalize_boxes_for_tracking(
    image_shapes, outputs, width, height, batch_size, gp=None, verbose=False
):
    """Normalize boxes for tracking input format - xyxy with confidence score.

    Args:
        image_shapes (list): List of image shapes.
        outputs (list): Predicted bounding boxes in xyxy format in YOLO pixel space.
        width (float): Image width.
        height (float): Image height.
        batch_size (int): Batch size.
        gp (dict): Grid parameters.
        verbose (bool): Verbose.

    Returns:
        Predictions containing bounding boxes in xyxy format in original image pixel space and original image width
        and height.
    """
    original_width = image_shapes[0][0][1][0][1]
    original_height = image_shapes[0][0][1][0][0]

    if gp:
        gp(0, "Formatting...")
    # keep predictions to feed them ordered into the Tracker
    # TODO: how to deal with large files?
    predictions = {}
    with tqdm(
        total=len(image_shapes) * batch_size,
        desc="Running formatting",
        ncols=0,
        disable=not verbose,
    ) as pbar:
        for batch_i, batch in enumerate(outputs):

            if gp:
                gp(batch_i / len(image_shapes), pbar.__str__())

            batch_shapes = image_shapes[batch_i]

            # Format results
            for si, pred in enumerate(batch):
                (image_shape, original_shape) = batch_shapes[si]
                # Clip boxes to image bounds and resize to input shape
                clip_boxes(pred, (height, width))
                box = pred[:, :4].clone()  # xyxy in yolo pixel space
                confs = pred[:, 4].clone().tolist()
                box = scale_boxes(
                    image_shape, box, original_shape[0], original_shape[1]
                )  # xyxy in original pixel space

                # confidence score currently not used by tracker; set to 1.0
                boxes = None
                if box.shape[0]:
                    original_width = original_shape[0][1]
                    original_height = original_shape[0][0]
                    do_norm = partial(
                        norm, w=original_shape[0][1], h=original_shape[0][0]
                    )
                    normed = list((map(do_norm, box[:, :4].tolist())))
                    boxes = np.stack([[*bb, conf] for bb, conf in zip(normed, confs)])

                frame_num = (batch_i, si)
                predictions[frame_num] = boxes

            pbar.update(1 * batch_size)

    return predictions, original_width, original_height


def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two bounding boxes in [x1, y1, x2, y2] format.

    Note: This implementation is adapted from [SORT](https://github.com/abewley/sort).
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def convert_bbox_to_z(bbox):
    """
    Converts a bounding box [x1, y1, x2, y2] to [x, y, s, r].

    Note: This implementation is adapted from [SORT](https://github.com/abewley/sort).
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right

    Note: This implementation is adapted from [SORT](https://github.com/abewley/sort).
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


def non_max_suppression(
    prediction,
    pix2width,
    conf,
    iou,
    max_length,
    max_det=300,
    max_nms=30000,
    redundant=True,
    merge=False,
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    NOTE: SIMPLIFIED FOR SINGLE CLASS DETECTION. Modified from yolov5/utils/general.py

    Args:
        prediction: Inference results containing bounding boxes in xyxy format relative to YOLO pixel space
        pix2width: Pixel-to-meter conversion factor. Used to filter out bounding boxes that are too small or represent
            fish below a size threshold.
        conf: NMS confidence score
        iou: NMS iou score
        max_length: Maximum fish length
        max_det: Maximum number of detections
        max_nms: Maximum number of boxes into torchvision.ops.nms()
        redundant: Require redundant detections
        merge: Use merge-NMS

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls] where xyxy is relative to YOLO pixel space
    """

    # Checks
    assert (
        0 <= conf <= 1
    ), f"Invalid Confidence threshold {conf}, valid values are between 0.0 and 1.0"
    assert 0 <= iou <= 1, f"Invalid IoU {iou}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    xc = prediction[..., 4] > conf  # candidates

    # width filter
    width = prediction[..., 2] * pix2width
    if max_length > 0:
        wc = width < max_length
    else:
        # If max_length is 0, ignore
        wc = width > max_length

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for image_idx, detections in enumerate(prediction):  # image index, image inference

        # Keep boxes that pass confidence threshold
        detections = detections[xc[image_idx] * wc[image_idx]]  # confidence

        # If none remain process next image
        if not detections.shape[0]:
            continue

        # Compute conf
        detections[:, 5:] *= detections[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            detections[:, :4]
        )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = detections[:, 6:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        confs, j = detections[:, 5:6].max(1, keepdim=True)
        detections = torch.cat((box, confs, j.float(), mask), 1)[confs.view(-1) > conf]

        # Check shape
        n = detections.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        detections = detections[
            detections[:, 4].argsort(descending=True)[:max_nms]
        ]  # sort by confidence and remove excess boxes

        # Batched NMS
        boxes = detections[:, :4]  # boxes (offset by class), scores
        scores = detections[:, 4]
        kept_idx = torchvision.ops.nms(boxes, scores, iou)  # NMS

        kept_idx = kept_idx[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[kept_idx], boxes) > iou  # iou matrix
            weights = iou * scores[None]  # box weights
            detections[kept_idx, :4] = torch.mm(
                weights, detections[:, :4]
            ).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                kept_idx = kept_idx[iou.sum(1) > 1]  # require redundancy

        output[image_idx] = detections[kept_idx]
        if mps:
            output[image_idx] = output[image_idx].to(device)

    return output


def run_nms(
    pred_bboxes,
    image_meter_width,
    image_pixel_width,
    batch_size,
    nms_config: NMSConfig = NMSConfig(),
    gp=None,
    verbose=False,
):
    """Run NMS on inference results to reject overlapping detections.

    Args:
        pred_bboxes: Inference results containing bounding boxes in xyxy format relative to YOLO pixel space.
        image_meter_width: Width of image in meters.
        image_pixel_width: Width of image in pixels.
        batch_size: Batch size.
        nms_config: NMSConfig
        gp: GPU parameters
        verbose: bool
    """

    # width filter
    pix2width = image_meter_width / image_pixel_width

    if gp:
        gp(0, "Suppression...")
    outputs = []
    with tqdm(
        total=len(pred_bboxes) * batch_size,
        desc="Running suppression",
        ncols=0,
        disable=not verbose,
    ) as pbar:
        for batch_idx, inf_out in enumerate(pred_bboxes):

            if gp:
                gp(batch_idx / len(pred_bboxes), pbar.__str__())

            with torch.no_grad():
                output = non_max_suppression(
                    inf_out,
                    pix2width,
                    conf=nms_config.conf,
                    iou=nms_config.iou,
                    max_length=nms_config.target_size.max_length,
                    max_det=nms_config.max_det,
                    max_nms=nms_config.max_nms,
                    redundant=nms_config.redundant,
                    merge=nms_config.merge,
                )

            outputs.append(output)
            pbar.update(1 * batch_size)

    return outputs


class NMSProcessor:
    """Handles low/high confidence NMS and normalization for tracking."""

    def __init__(self, nms_config, metadata, batch_size: int):
        self.nms_config = nms_config or NMSConfig()
        self.metadata = metadata
        self.batch_size = batch_size

    def run(self, batch_pred: torch.Tensor, batch_shapes) -> Tuple[Dict, Dict]:
        """Apply NMS at low (0.1) and high (0.3) confidence thresholds."""
        low_conf, high_conf = 0.1, 0.3

        batch_pred = batch_pred.cpu()
        (img_height, img_width) = batch_shapes[0][0]

        # Low confidence NMS
        self.nms_config.conf = low_conf
        low_output = run_nms(
            [batch_pred],
            self.metadata.image_meter_width,
            img_width,
            self.batch_size,
            self.nms_config,
        )
        low_preds, *_ = normalize_boxes_for_tracking(
            [batch_shapes],
            low_output,
            width=img_width,
            height=img_height,
            batch_size=self.batch_size,
        )

        # High confidence NMS
        self.nms_config.conf = high_conf
        high_output = run_nms(
            [batch_pred],
            self.metadata.image_meter_width,
            img_width,
            self.batch_size,
            self.nms_config,
        )
        high_preds, *_ = normalize_boxes_for_tracking(
            [batch_shapes],
            high_output,
            width=img_width,
            height=img_height,
            batch_size=self.batch_size,
        )

        return low_preds, high_preds

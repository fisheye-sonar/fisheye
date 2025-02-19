import torch
import torchvision
from tqdm import tqdm
from yolov5.utils.general import xywh2xyxy
from yolov5.utils.metrics import box_iou

from fisheye.dataclasses import YOLOv5ModelConfig


def non_max_suppression(
    prediction, pix2width, config, max_nms=30000, redundant=True, merge=False
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    NOTE: SIMPLIFIED FOR SINGLE CLASS DETECTION. Modified from yolov5/utils/general.py

    :arg
        prediction: Inference results
        pix2width: Pixel size
        config: Detector's configuration
        max_nms: Maximum number of boxes into torchvision.ops.nms()
        redundant: Require redundant detections
        merge: Use merge-NMS

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert (
        0 <= config.conf <= 1
    ), f"Invalid Confidence threshold {config.conf}, valid values are between 0.0 and 1.0"
    assert (
        0 <= config.iou <= 1
    ), f"Invalid IoU {config.iou}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    xc = prediction[..., 4] > config.conf  # candidates

    # width filter
    width = prediction[..., 2] * pix2width
    if config.max_length > 0:
        wc = width < config.max_length
    else:
        # If max_length is 0, ignore
        wc = width > config.max_length

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
        conf, j = detections[:, 5:6].max(1, keepdim=True)
        detections = torch.cat((box, conf, j.float(), mask), 1)[
            conf.view(-1) > config.conf
        ]

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
        kept_idx = torchvision.ops.nms(boxes, scores, config.iou)  # NMS

        kept_idx = kept_idx[: config.max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[kept_idx], boxes) > config.iou  # iou matrix
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
    inference,
    image_meter_width,
    image_pixel_width,
    model_config: YOLOv5ModelConfig,
    gp=None,
    verbose=True,
):
    """Run NMS on inference results to reject overlapping detections."""

    # width filter
    pix2width = image_meter_width / image_pixel_width

    if gp:
        gp(0, "Suppression...")
    # TODO: how to deal with large files?
    outputs = []
    with tqdm(
        total=len(inference) * model_config.batch_size,
        desc="Running suppression",
        ncols=0,
        disable=not verbose,
    ) as pbar:
        for batch_idx, inf_out in enumerate(inference):

            if gp:
                gp(batch_idx / len(inference), pbar.__str__())

            with torch.no_grad():
                output = non_max_suppression(inf_out, pix2width, model_config)

            outputs.append(output)
            pbar.update(1 * model_config.batch_size)

    return outputs

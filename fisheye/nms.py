import torch
import torchvision
from tqdm import tqdm
from yolov5.utils.general import xywh2xyxy
from yolov5.utils.metrics import box_iou

from fisheye.dataclasses import YOLOv5ModelConfig


def non_max_suppression(prediction, image_meter_width, image_pixel_width, config):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    NOTE: SIMPLIFIED FOR SINGLE CLASS DETECTION. Modified from yolov5/utils/general.py

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
    pix2width = image_meter_width / image_pixel_width
    width = prediction[..., 2] * pix2width
    if config.max_length > 0:
        wc = width < config.max_length
    else:
        # If max_length is 0, ignore
        wc = width > config.max_length

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference

        # Keep boxes that pass confidence threshold
        x = x[xc[xi] * wc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            x[:, :4]
        )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, 6:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:6].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[
            x[:, 4].argsort(descending=True)[:max_nms]
        ]  # sort by confidence and remove excess boxes

        # Batched NMS
        boxes = x[:, :4]  # boxes (offset by class), scores
        scores = x[:, 4]
        i = torchvision.ops.nms(boxes, scores, config.iou)  # NMS

        i = i[: config.max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > config.iou  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)

        logging = False

    return output


def run_nms(
    inference,
    image_meter_width,
    image_pixel_width,
    config: YOLOv5ModelConfig,
    gp=None,
    verbose=True,
):
    """Run NMS on inference results to reject overlapping detections."""

    if gp:
        gp(0, "Suppression...")
    # TODO: how to deal with large files?
    outputs = []
    with tqdm(
        total=len(inference) * config.batch_size,
        desc="Running suppression",
        ncols=0,
        disable=not verbose,
    ) as pbar:
        for batch_i, inf_out in enumerate(inference):

            if gp:
                gp(batch_i / len(inference), pbar.__str__())

            with torch.no_grad():
                output = non_max_suppression(
                    inf_out, image_meter_width, image_pixel_width, config
                )

            outputs.append(output)
            pbar.update(1 * config.batch_size)

    return outputs

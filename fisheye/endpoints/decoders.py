"""Model-specific outputs mapped into the common endpoint result contract."""

from __future__ import annotations

from .types import EndpointResult


def select_detection(detections, target):
    """Reference match: positive IoU, finite rows, then highest IoU/confidence."""
    import torch

    if len(detections) == 0:
        return None
    target = target.to(detections.device)
    boxes = detections[:, :4]
    intersection = (
        (
            torch.minimum(boxes[:, 2:], target[2:])
            - torch.maximum(boxes[:, :2], target[:2])
        )
        .clamp(min=0)
        .prod(1)
    )
    area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1)
    target_area = (target[2:] - target[:2]).clamp(min=0).prod()
    iou = intersection / (area + target_area - intersection).clamp(min=1e-9)
    candidates = [
        i
        for i in range(len(detections))
        if iou[i] > 0 and torch.isfinite(detections[i]).all()
    ]
    if not candidates:
        return None
    return (
        detections[
            max(candidates, key=lambda i: (float(iou[i]), float(detections[i, 4])))
        ]
        .detach()
        .cpu()
    )


def _result(sample, points, scores):
    import torch

    full = sample.transform.to_full(points.detach().cpu())
    if not torch.isfinite(full).all():
        return EndpointResult.from_query(
            sample.query,
            "no_prediction",
            transform=sample.transform,
            reason={"code": "nonfinite_endpoints"},
        )
    order = torch.argsort(full[:, 0], stable=True)
    full = full[order]
    for key in (
        "keypoint_confidence",
        "endpoint_entropy",
        "endpoint_peak",
        "endpoint_scores",
    ):
        if key in scores:
            scores[key] = [scores[key][int(i)] for i in order]
    return EndpointResult.from_query(
        sample.query,
        endpoints_xy_px=tuple(tuple(float(v) for v in p) for p in full),
        length_pixels=float(torch.linalg.vector_norm(full[1] - full[0])),
        scores=scores,
        transform=sample.transform,
    )


class PoseDecoder:
    component_id = "yolo_pose"
    version = "1"
    input_schema = "pose_candidates.v1"

    def __init__(self, candidate_confidence=0.001, iou_threshold=0.7, max_det=300):
        if (
            not 0 <= candidate_confidence <= 1
            or not 0 < iou_threshold <= 1
            or max_det < 1
        ):
            raise ValueError("Invalid YOLO candidate/NMS settings")
        self.candidate_confidence = candidate_confidence
        self.iou_threshold = iou_threshold
        self.max_det = max_det

    def decode(self, output, batch):
        from ultralytics.utils.nms import non_max_suppression

        candidates = non_max_suppression(
            output.prediction,
            conf_thres=self.candidate_confidence,
            iou_thres=self.iou_threshold,
            nc=1,
            end2end=output.end2end,
            max_det=self.max_det,
        )
        if len(candidates) != len(batch.samples):
            raise ValueError("YOLO output count differs from prepared sample count")
        results = []
        for rows, sample in zip(candidates, batch.samples):
            selected = select_detection(rows, sample.target_box)
            if selected is None:
                results.append(
                    EndpointResult.from_query(
                        sample.query,
                        "no_prediction",
                        transform=sample.transform,
                        scores={
                            "source": "yolo_detection_confidence",
                            "candidate_count": len(rows),
                        },
                        reason={"code": "no_overlapping_pose_candidate"},
                    )
                )
                continue
            points = selected[6:].reshape(2, 3)
            results.append(
                _result(
                    sample,
                    points[:, :2],
                    {
                        "source": "yolo_detection_confidence",
                        "yolo_confidence": float(selected[4]),
                        "keypoint_confidence": points[:, 2].tolist(),
                        "candidate_count": len(rows),
                        "measurement_reliability_calibrated": False,
                    },
                )
            )
        return results


def target_heatmap_distribution(logits, support):
    """Exact masked softmax/grid convention from the trained target heatmap model."""
    import torch
    import torch.nn.functional as F

    h, w = logits.shape[-2:]
    valid = F.interpolate(support.float(), (h, w), mode="nearest").bool()
    if not valid.flatten(2).any(-1).all():
        raise ValueError("No real-image support in heatmap")
    log_probs = logits.float().masked_fill(~valid, -1e9).flatten(2).log_softmax(-1)
    yy, xx = torch.meshgrid(
        (torch.arange(h, device=logits.device) + 0.5) / h,
        (torch.arange(w, device=logits.device) + 0.5) / w,
        indexing="ij",
    )
    grid = torch.stack((xx, yy), -1).reshape(-1, 2)
    return log_probs, grid, valid


class TargetHeatmapDecoder:
    component_id = "target_heatmap"
    version = "1"
    input_schema = "target_heatmap_logits.v1"

    def decode(self, output, batch):
        import torch

        if output.ndim != 4 or output.shape[:2] != (len(batch.samples), 2):
            raise ValueError("Target heatmap output must be B x 2 x H x W")
        probabilities, grid, _ = target_heatmap_distribution(
            output, batch.support.to(output.device)
        )
        points = probabilities.exp() @ grid
        entropy = -(probabilities.exp() * probabilities).sum(-1)
        # The model produces coordinates normalized to its input canvas, not its
        # stride-four output grid; use the actual collation canvas dimensions.
        height, width = batch.images.shape[-2:]
        points = points * points.new_tensor([width, height])
        return [
            _result(
                sample,
                p,
                {
                    "source": "target_heatmap",
                    "endpoint_entropy": e.detach().cpu().tolist(),
                    "measurement_reliability_calibrated": False,
                    "availability_is_not_detection_or_measurement_coverage": True,
                },
            )
            for p, e, sample in zip(points, entropy, batch.samples)
        ]


def native_map_to_points(
    heatmap, differentiable=False, round_to_integer=False, power=10
):
    """Reference mapTokpt: raw-map power expectation or integer-grid argmax."""
    import torch

    if differentiable:
        weighted = heatmap**power
        h, w = weighted.shape[-2:]
        sy, sx = weighted.sum(3), weighted.sum(2)
        y = (torch.arange(h, device=heatmap.device) * sy).sum(2) / sy.sum(2)
        x = (torch.arange(w, device=heatmap.device) * sx).sum(2) / sx.sum(2)
    else:
        h, w = heatmap.shape[-2:]
        indices = heatmap.reshape(*heatmap.shape[:2], -1).argmax(-1)
        y, x = (indices // w).float(), (indices % w).float()
    points = torch.stack([x, y], -1)
    return points.round() if round_to_integer else points


class NativeHeatmapDecoder:
    component_id = "native_heatmap"
    version = "1"
    input_schema = "native_heatmap.v1"

    def __init__(
        self,
        differentiable=False,
        round_to_integer=False,
        power=10,
        decode_crop_padding=None,
    ):
        self.differentiable, self.round_to_integer = differentiable, round_to_integer
        self.power, self.decode_crop_padding = power, decode_crop_padding

    def decode(self, output, batch):
        import torch

        if (
            output.shape[:2] != (len(batch.samples), 2)
            or output.shape[-2:] != batch.images.shape[-2:]
        ):
            raise ValueError(
                "Native heatmap output must retain the input B x 2 x H x W canvas"
            )
        results = []
        for i, sample in enumerate(batch.samples):
            h, w = sample.transform.crop_hw
            if tuple(sample.transform.gain_xy) != (1.0, 1.0) or tuple(
                sample.transform.pad_xy
            ) != (0, 0):
                raise ValueError(
                    "Native heatmap decoder requires unscaled, unshifted crop pixels"
                )
            left = top = 0
            right, bottom = w, h
            if self.decode_crop_padding is not None:
                padding = self.decode_crop_padding
                target = sample.target_box.cpu()
                left = max(0, int(target[0]) - padding)
                top = max(0, int(target[1]) - padding)
                right = min(w, int(target[2]) + padding)
                bottom = min(h, int(target[3]) + padding)
            maps = output[i : i + 1, :, top:bottom, left:right]
            points = native_map_to_points(
                maps, self.differentiable, self.round_to_integer, self.power
            )[0]
            points = points + points.new_tensor([left, top])
            peaks = maps.flatten(2).max(-1).values[0]
            results.append(
                _result(
                    sample,
                    points,
                    {
                        "source": "native_heatmap_raw_peak",
                        "endpoint_peak": peaks.detach().cpu().tolist(),
                        "heatmap_peak_mean": float(peaks.mean()),
                        "measurement_reliability_calibrated": False,
                    },
                )
            )
        return results


DECODERS = {
    "yolo_pose": PoseDecoder,
    "target_heatmap": TargetHeatmapDecoder,
    "native_heatmap": NativeHeatmapDecoder,
}


def create_decoder(config):
    values = dict(config)
    name = values.pop("name")
    try:
        decoder = DECODERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown endpoint decoder {name!r}; available: {sorted(DECODERS)}"
        ) from exc
    return decoder(**values)

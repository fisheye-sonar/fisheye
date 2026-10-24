"""Persistent model execution without hidden preprocessing or network access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ULTRALYTICS_VERSION = "8.4.51"
BACKEND_SCHEMAS = {
    "yolo_pose": "pose_candidates.v1",
    "yolo_target_heatmap": "target_heatmap_logits.v1",
    "unet_heatmap": "native_heatmap.v1",
    "cnn_heatmap": "native_heatmap.v1",
}


@dataclass(frozen=True)
class PoseOutput:
    prediction: Any
    end2end: bool


def descriptor_from_preprocessing(config):
    from .types import InputDescriptor

    offsets = tuple(config["temporal_offsets"])
    target_mask = bool(config.get("target_mask", False))
    spatial = config.get("spatial_mode", "letterbox")
    fixed_hw = None
    if spatial == "letterbox":
        size = config.get("image_size", 640)
        fixed_hw = (size, size) if isinstance(size, int) else tuple(size)
    return InputDescriptor(
        channels=len(offsets) + int(target_mask),
        representations=(config.get("representation", "bgs_signed_u8"),),
        temporal_offsets=offsets,
        target_mask=target_mask,
        spatial_mode=spatial,
        stride=int(config.get("stride", 32)),
        fixed_hw=fixed_hw,
    )


def require_ultralytics():
    import ultralytics

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Endpoint YOLO models require ultralytics=={ULTRALYTICS_VERSION}; "
            f"found {ultralytics.__version__}. Install the pinned project dependencies."
        )


def build_model(payload):
    """Construct solely from embedded architecture/config; never load pretrained weights."""
    import copy
    import torch.nn as nn

    backend = payload["backend"]
    config = payload["config"]
    channels = payload["input_channels"]
    if backend in ("yolo_pose", "yolo_target_heatmap"):
        require_ultralytics()
        from ultralytics.nn.tasks import PoseModel

        # Construct the unconditioned stem first: checkpoint YAML describes that
        # architecture, while target variants add exactly one conditioning mask.
        masked = config.get("experiment", {}).get("mode") in (
            "target_pose",
            "target_heatmap",
        )
        pose = PoseModel(
            copy.deepcopy(payload["architecture"]),
            ch=channels - int(masked),
            nc=1,
            data_kpt_shape=[2, 3],
            verbose=False,
        )
        pose.names = {0: "fish"}
        pose.kpt_shape = [2, 3]
        if masked:
            old = pose.model[0].conv
            pose.model[0].conv = nn.Conv2d(
                old.in_channels + 1,
                old.out_channels,
                old.kernel_size,
                old.stride,
                old.padding,
                dilation=old.dilation,
                bias=old.bias is not None,
            )
        if backend == "yolo_target_heatmap":
            from .vendor.target_heatmap import TargetHeatmapModel

            return TargetHeatmapModel(pose, config)
        return pose
    if backend in ("unet_heatmap", "cnn_heatmap"):
        from .vendor.native_heatmap import HeatmapCNN, UNetHeatmap

        if backend == "cnn_heatmap":
            return HeatmapCNN(in_ch=channels)
        return UNetHeatmap(
            in_ch=channels, use_double_conv=config["model"]["unet_double_conv"]
        )
    raise ValueError(
        f"Unknown endpoint backend {backend!r}; available: {sorted(BACKEND_SCHEMAS)}"
    )


class TorchModelBackend:
    """One loaded model per worker. Calls are intentionally not thread safe."""

    version = "1"

    def __init__(self, payload, preprocessing, *, device="cpu"):
        import torch

        self.component_id = payload["backend"]
        self.output_schema = BACKEND_SCHEMAS[self.component_id]
        self.input_descriptor = descriptor_from_preprocessing(preprocessing)
        if self.input_descriptor.channels != payload["input_channels"]:
            raise ValueError(
                "Preprocessing channel count differs from checkpoint input channels"
            )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Endpoint device {device!r} requested, but CUDA is unavailable"
            )
        self.model = build_model(payload)
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.to(device=self.device, dtype=torch.float32).eval()

    def predict(self, batch):
        import torch

        images = batch.images
        if images.ndim != 4 or images.shape[1] != self.input_descriptor.channels:
            raise ValueError(
                f"Expected BCHW input with {self.input_descriptor.channels} channels"
            )
        with torch.inference_mode():
            output = self.model(images.to(device=self.device, dtype=torch.float32))
        if self.component_id == "yolo_pose":
            return PoseOutput(output, bool(self.model.end2end))
        return output

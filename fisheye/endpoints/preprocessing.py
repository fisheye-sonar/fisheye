"""Reference crop/stack transforms, replaceable without touching model code.

Geometry and YOLO transforms reproduce length_annotator's
model/predict_clip_pipeline.py and model/yolo_pose/{pipeline,experiment_data}.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Integral
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .temporal import TrackTemporalSelector
from .types import (
    CoordinateTransform,
    EndpointQuery,
    FrameStore,
    InputDescriptor,
    PreparedBatch,
    PreparedSample,
    QueryValidationError,
    SpatialPreparedImage,
    SpatialTransform,
    TemporalStacker,
)


@dataclass(frozen=True)
class PreprocessingConfig:
    representation: str = "bgs_signed_u8"
    temporal_offsets: tuple[int, ...] = (0,)
    crop_expansion: float = 1.0
    spatial_mode: str = "letterbox"
    image_size: int | tuple[int, int] | None = 640
    stride: int = 32
    target_mask: bool = False
    pad_value: float = 114 / 255
    batch_size: int = 1
    native_canvas: str = "stride"

    def __post_init__(self):
        object.__setattr__(self, "temporal_offsets", tuple(self.temporal_offsets))
        TrackTemporalSelector(offsets=self.temporal_offsets)
        if not self.representation:
            raise ValueError("A named stored image representation is required")
        if not math.isfinite(self.crop_expansion) or self.crop_expansion < 0:
            raise ValueError("Crop expansion must be finite and nonnegative")
        if self.spatial_mode not in ("letterbox", "native"):
            raise ValueError("Spatial mode must be letterbox or native")
        for name in ("stride", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.pad_value) or not 0 <= self.pad_value <= 1:
            raise ValueError("Padding value must lie in [0, 1]")
        if self.native_canvas not in ("stride", "batch_max", "none"):
            raise ValueError("Native canvas must be stride, batch_max, or none")
        if self.spatial_mode == "letterbox":
            if self.image_size is None:
                raise ValueError("Letterbox requires image_size")
            size = (
                (self.image_size, self.image_size)
                if isinstance(self.image_size, Integral)
                else tuple(self.image_size)
            )
            if len(size) != 2 or any(
                isinstance(x, bool) or not isinstance(x, Integral) or x < 1
                for x in size
            ):
                raise ValueError(
                    "image_size must be a positive integer or height/width pair"
                )
            if any(x % self.stride for x in size):
                raise ValueError(
                    "Letterbox dimensions must be divisible by model stride"
                )
            object.__setattr__(self, "image_size", tuple(int(x) for x in size))

    @property
    def descriptor(self):
        return InputDescriptor(
            channels=len(self.temporal_offsets) + int(self.target_mask),
            representations=(self.representation,),
            temporal_offsets=self.temporal_offsets,
            target_mask=self.target_mask,
            spatial_mode=self.spatial_mode,
            stride=self.stride,
            fixed_hw=self.image_size if self.spatial_mode == "letterbox" else None,
        )

    def to_dict(self):
        return asdict(self)


def expanded_crop(box, frame_hw, expansion: float = 1.0):
    """Outward integer bounds, then percent padding on each integer extent."""
    h, w = frame_hw
    x1, y1, x2, y2 = map(float, box)
    left, top, right, bottom = (
        math.floor(x1),
        math.floor(y1),
        math.ceil(x2),
        math.ceil(y2),
    )
    padx, pady = int((right - left) * expansion), int((bottom - top) * expansion)
    return (
        max(0, left - padx),
        max(0, top - pady),
        min(w, right + padx),
        min(h, bottom + pady),
    )


class ChannelTemporalStacker:
    """Reference chronological channels and uint8-to-[0, 1] conversion."""

    component_id = "temporal_channels.v1"

    def describe(self, source: InputDescriptor):
        return source

    def stack(self, crops: list[np.ndarray]):
        return torch.from_numpy(np.stack(crops)).to(torch.float32).div_(255)


class ReferenceSpatialTransform:
    """Exact bilinear letterbox or unscaled right/bottom native padding."""

    component_id = "crop_spatial.v1"

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def describe(self, source: InputDescriptor):
        return source

    def apply(self, image, origin_xy, frame_hw):
        _, h, w = image.shape
        if self.config.spatial_mode == "letterbox":
            canvas_h, canvas_w = self.config.image_size
            scale = min(canvas_h / h, canvas_w / w)
            nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
            pad_top, pad_left = (canvas_h - nh) // 2, (canvas_w - nw) // 2
            resized = F.interpolate(
                image[None], (nh, nw), mode="bilinear", align_corners=False
            )[0]
            image = F.pad(
                resized,
                (pad_left, canvas_w - nw - pad_left, pad_top, canvas_h - nh - pad_top),
                value=self.config.pad_value,
            )
        else:
            nh, nw, pad_top, pad_left = h, w, 0, 0
            stride = self.config.stride if self.config.native_canvas == "stride" else 1
            canvas_h, canvas_w = h + (-h) % stride, w + (-w) % stride
            image = F.pad(
                image, (0, canvas_w - w, 0, canvas_h - h), value=self.config.pad_value
            )
        transform = CoordinateTransform(
            tuple(origin_xy),
            (nw / w, nh / h),
            (pad_left, pad_top),
            (h, w),
            (canvas_h, canvas_w),
            tuple(frame_hw),
        )
        support = torch.zeros((1, canvas_h, canvas_w), dtype=torch.bool)
        support[:, pad_top : pad_top + nh, pad_left : pad_left + nw] = True
        return SpatialPreparedImage(image, support, transform)


class CropStackPreprocessor:
    component_id = "crop_stack.v1"

    def __init__(
        self,
        config: PreprocessingConfig | dict | None = None,
        *,
        frame_stacker: TemporalStacker | None = None,
        spatial_transform: SpatialTransform | None = None,
    ):
        self.config = (
            config
            if isinstance(config, PreprocessingConfig)
            else PreprocessingConfig(**(config or {}))
        )
        self.frame_stacker = (
            frame_stacker if frame_stacker is not None else ChannelTemporalStacker()
        )
        self.spatial_transform = (
            spatial_transform
            if spatial_transform is not None
            else ReferenceSpatialTransform(self.config)
        )
        self.descriptor = self.spatial_transform.describe(
            self.frame_stacker.describe(self.config.descriptor)
        )

    def validate_query(self, frame_store: FrameStore, query: EndpointQuery):
        if query.clip_id != frame_store.clip_id:
            raise QueryValidationError(
                "clip_mismatch", "Query and frame store belong to different clips"
            )
        if (
            isinstance(query.frame_id, bool)
            or not isinstance(query.frame_id, Integral)
            or any(
                isinstance(fid, bool) or not isinstance(fid, Integral)
                for fid in query.context_frame_ids
            )
        ):
            raise QueryValidationError(
                "frame_identity", "Frame identities must be integers"
            )
        try:
            box = np.asarray(query.bbox_xyxy_px, dtype=float)
        except (TypeError, ValueError) as exc:
            raise QueryValidationError(
                "invalid_box", "Expected four finite pixel coordinates"
            ) from exc
        if (
            box.shape != (4,)
            or not np.isfinite(box).all()
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            raise QueryValidationError(
                "invalid_box", "Target box must be finite and have positive area"
            )
        if len(query.context_frame_ids) != len(self.config.temporal_offsets):
            raise QueryValidationError(
                "temporal_count",
                "Query temporal count does not match preprocessing profile",
            )
        if len(query.context_repeated) != len(query.context_frame_ids):
            raise QueryValidationError(
                "temporal_markers",
                "Every context position requires a repetition marker",
            )
        center = self.config.temporal_offsets.index(0)
        if query.context_frame_ids[center] != query.frame_id:
            raise QueryValidationError(
                "temporal_center", "Zero-offset context must be the query center frame"
            )
        shapes = set()
        for frame_id in query.context_frame_ids:
            try:
                if hasattr(frame_store, "frame_shape"):
                    shape = frame_store.frame_shape(
                        frame_id, self.config.representation
                    )
                else:
                    frame = frame_store.get(frame_id, self.config.representation)
                    if frame.dtype != np.uint8:
                        raise QueryValidationError(
                            "image_dtype", "Stored images must be HW uint8"
                        )
                    shape = frame.shape
            except KeyError as exc:
                raise QueryValidationError(
                    "missing_frame", str(exc), status="missing_frame"
                ) from exc
            shapes.add(tuple(shape))
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise QueryValidationError(
                "image_shape", "Temporal images must share the original HW dimensions"
            )
        frame_hw = next(iter(shapes))
        crop = (
            query.crop_xyxy_px
            if query.crop_xyxy_px is not None
            else expanded_crop(box, frame_hw, self.config.crop_expansion)
        )
        if (
            len(crop) != 4
            or any(isinstance(x, bool) or not isinstance(x, Integral) for x in crop)
            or not 0 <= crop[0] < crop[2] <= frame_hw[1]
            or not 0 <= crop[1] < crop[3] <= frame_hw[0]
        ):
            raise QueryValidationError(
                "invalid_crop",
                "Crop must be a nonempty integer rectangle within the original frame",
            )
        if box[2] <= 0 or box[3] <= 0 or box[0] >= frame_hw[1] or box[1] >= frame_hw[0]:
            raise QueryValidationError(
                "box_outside_frame", "Target box does not intersect the frame"
            )
        if (
            box[2] <= crop[0]
            or box[3] <= crop[1]
            or box[0] >= crop[2]
            or box[1] >= crop[3]
        ):
            raise QueryValidationError(
                "box_outside_crop", "Target box does not intersect the selected crop"
            )
        return tuple(int(x) for x in crop), frame_hw

    def _prepare_sample(self, store, query):
        crop, frame_hw = self.validate_query(store, query)
        left, top, right, bottom = crop
        # Slice before allocating float data: no full-frame float tensors.
        arrays = [
            np.array(
                store.get(fid, self.config.representation)[top:bottom, left:right],
                copy=True,
            )
            for fid in query.context_frame_ids
        ]
        image = self.frame_stacker.stack(arrays)
        if image.ndim != 3 or tuple(image.shape[-2:]) != (bottom - top, right - left):
            raise ValueError(
                "Temporal stacker must preserve the cropped HW pixel geometry"
            )
        spatial = self.spatial_transform.apply(image, (left, top), tuple(frame_hw))
        image, support, transform = spatial.image, spatial.support, spatial.transform
        h, w = transform.crop_hw
        canvas_h, canvas_w = transform.canvas_hw
        pad_left, pad_top = transform.pad_xy
        nh, nw = round(h * transform.gain_xy[1]), round(w * transform.gain_xy[0])
        target = torch.tensor(query.bbox_xyxy_px, dtype=torch.float32).reshape(2, 2)
        # Match reference candidate matching at clipped frame/crop edges while
        # retaining the original target box on the public query/result.
        local_target = target - target.new_tensor(transform.origin_xy)
        local_target[:, 0].clamp_(0, w)
        local_target[:, 1].clamp_(0, h)
        model_target = local_target * local_target.new_tensor(
            transform.gain_xy
        ) + local_target.new_tensor(transform.pad_xy)
        if self.config.target_mask:
            # The trained conditioning-mask implementation round-trips the
            # already transformed target through (p-pad)/gain. Preserve its
            # float32 rounding before floor/ceil, including integer edges.
            local = (
                model_target - model_target.new_tensor(transform.pad_xy)
            ) / model_target.new_tensor(transform.gain_xy)
            local[:, 0].clamp_(0, w)
            local[:, 1].clamp_(0, h)
            low, high = local.min(0).values, local.max(0).values
            x1, y1 = torch.floor(low).long().tolist()
            x2, y2 = torch.ceil(high).long().tolist()
            mask = torch.zeros((1, h, w), dtype=torch.float32)
            mask[:, y1 : max(y1 + 1, y2), x1 : max(x1 + 1, x2)] = 1
            mask = F.interpolate(mask[None], (nh, nw), mode="nearest")[0]
            mask = F.pad(
                mask,
                (pad_left, canvas_w - nw - pad_left, pad_top, canvas_h - nh - pad_top),
                value=0,
            )
            image = torch.cat((image, mask), dim=0)
        sample = PreparedSample(query, transform, model_target.reshape(4))
        return image, support, sample

    def _collate(self, prepared):
        height = max(item[0].shape[-2] for item in prepared)
        width = max(item[0].shape[-1] for item in prepared)
        if self.config.native_canvas == "none" and any(
            item[0].shape[-2:] != (height, width) for item in prepared
        ):
            raise ValueError(
                "native_canvas=none requires identical shapes or batch_size=1"
            )
        images, supports, samples = [], [], []
        for image, support, sample in prepared:
            h, w = image.shape[-2:]
            image = F.pad(
                image, (0, width - w, 0, height - h), value=self.config.pad_value
            )
            if self.config.target_mask:
                # Image fill is 114/255 where configured; unknown mask fill is zero.
                image[-1, h:, :] = 0
                image[-1, :, w:] = 0
            images.append(image)
            supports.append(F.pad(support, (0, width - w, 0, height - h), value=False))
            samples.append(
                replace(
                    sample,
                    transform=replace(sample.transform, canvas_hw=(height, width)),
                )
            )
        return PreparedBatch(
            torch.stack(images), torch.stack(supports), samples, self.descriptor
        )

    def prepare(self, frame_store: FrameStore, queries: Iterable[EndpointQuery]):
        pending = []
        for query in queries:
            pending.append(self._prepare_sample(frame_store, query))
            if len(pending) == self.config.batch_size:
                yield self._collate(pending)
                pending = []
        if pending:
            yield self._collate(pending)

    def to_dict(self):
        return {
            "component_id": self.component_id,
            "config": self.config.to_dict(),
            "descriptor": asdict(self.descriptor),
            "frame_stacker": {"component_id": self.frame_stacker.component_id},
            "spatial_transform": {"component_id": self.spatial_transform.component_id},
        }

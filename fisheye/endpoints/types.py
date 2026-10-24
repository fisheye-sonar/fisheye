"""Public, model-independent endpoint inference contracts.

Coordinates use warped full-frame pixels and half-open crop bounds throughout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
import torch


@dataclass(frozen=True)
class InputDescriptor:
    channels: int
    layout: str = "BCHW"
    dtype: str = "float32"
    value_range: tuple[float, float] = (0.0, 1.0)
    representations: tuple[str, ...] = ("bgs_signed_u8",)
    temporal_offsets: tuple[int, ...] = (0,)
    target_mask: bool = False
    spatial_mode: str = "letterbox"
    stride: int = 32
    fixed_hw: tuple[int, int] | None = (640, 640)

    def __post_init__(self):
        for name in ("value_range", "representations", "temporal_offsets"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.fixed_hw is not None:
            object.__setattr__(self, "fixed_hw", tuple(self.fixed_hw))
        if self.channels < 1 or self.stride < 1:
            raise ValueError("Input channels and stride must be positive")

    def validate_compatible(self, required: InputDescriptor) -> None:
        for name in (
            "layout",
            "channels",
            "dtype",
            "value_range",
            "representations",
            "temporal_offsets",
            "target_mask",
            "spatial_mode",
            "stride",
            "fixed_hw",
        ):
            actual, expected = getattr(self, name), getattr(required, name)
            if actual != expected:
                raise ValueError(
                    f"Incompatible endpoint input {name}: preprocessor supplies "
                    f"{actual!r}, model requires {expected!r}"
                )


@dataclass(frozen=True)
class EndpointQuery:
    query_id: str
    clip_id: str
    frame_id: int
    track_id: int | None
    detection_id: str | None
    bbox_xyxy_px: tuple[float, float, float, float]
    context_frame_ids: tuple[int, ...]
    context_repeated: tuple[bool, ...]
    crop_xyxy_px: tuple[int, int, int, int] | None = None

    def __post_init__(self):
        for name in ("bbox_xyxy_px", "context_frame_ids", "context_repeated"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.crop_xyxy_px is not None:
            object.__setattr__(self, "crop_xyxy_px", tuple(self.crop_xyxy_px))


@dataclass(frozen=True)
class CoordinateTransform:
    origin_xy: tuple[float, float]
    gain_xy: tuple[float, float]
    pad_xy: tuple[int, int]
    crop_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    frame_hw: tuple[int, int] | None = None

    @property
    def origin(self):
        return self.origin_xy

    @property
    def gain(self):
        return self.gain_xy

    @property
    def pad(self):
        return self.pad_xy

    @property
    def crop_xyxy_px(self):
        x, y = self.origin_xy
        h, w = self.crop_hw
        return (int(x), int(y), int(x + w), int(y + h))

    def to_full(self, points):
        """Invert the exact rounded resize geometry, retaining tensor devices."""
        if torch.is_tensor(points):
            if not torch.is_floating_point(points):
                points = points.to(torch.float32)
            gain = points.new_tensor(self.gain_xy)
            pad = points.new_tensor(self.pad_xy)
            origin = points.new_tensor(self.origin_xy)
        else:
            points = np.asarray(points, dtype=np.float64)
            gain, pad, origin = self.gain_xy, self.pad_xy, self.origin_xy
        return (points - pad) / gain + origin

    def to_model(self, points):
        if torch.is_tensor(points):
            if not torch.is_floating_point(points):
                points = points.to(torch.float32)
            gain = points.new_tensor(self.gain_xy)
            pad = points.new_tensor(self.pad_xy)
            origin = points.new_tensor(self.origin_xy)
        else:
            points = np.asarray(points, dtype=np.float64)
            gain, pad, origin = self.gain_xy, self.pad_xy, self.origin_xy
        return (points - origin) * gain + pad


@dataclass
class PreparedSample:
    query: EndpointQuery
    transform: CoordinateTransform
    target_box: torch.Tensor


@dataclass
class PreparedBatch:
    images: torch.Tensor
    support: torch.Tensor
    samples: list[PreparedSample]
    descriptor: InputDescriptor

    @property
    def target_boxes(self):
        return torch.stack([sample.target_box for sample in self.samples])


@dataclass(frozen=True)
class EndpointResult:
    query_id: str
    clip_id: str
    frame_id: int
    track_id: int | None
    detection_id: str | None
    bbox_xyxy_px: tuple[float, float, float, float]
    status: str
    endpoints_xy_px: tuple[tuple[float, float], tuple[float, float]] | None = None
    length_pixels: float | None = None
    length_cm: float | None = None
    scores: Mapping[str, Any] = field(default_factory=dict)
    context_frame_ids: tuple[int, ...] = ()
    context_repeated: tuple[bool, ...] = ()
    transform: CoordinateTransform | None = None
    reason: Mapping[str, Any] | None = None
    flags: tuple[str, ...] = ()

    @classmethod
    def from_query(cls, query: EndpointQuery, status: str = "ok", **kwargs):
        return cls(
            query_id=query.query_id,
            clip_id=query.clip_id,
            frame_id=query.frame_id,
            track_id=query.track_id,
            detection_id=query.detection_id,
            bbox_xyxy_px=query.bbox_xyxy_px,
            context_frame_ids=query.context_frame_ids,
            context_repeated=query.context_repeated,
            status=status,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrameStore(Protocol):
    clip_id: str

    def get(self, frame_id: int, representation: str) -> np.ndarray: ...


class InputPreprocessor(Protocol):
    descriptor: InputDescriptor

    def prepare(
        self, frame_store: FrameStore, queries: Iterable[EndpointQuery]
    ) -> Iterable[PreparedBatch]: ...


class TemporalStacker(Protocol):
    component_id: str

    def describe(self, source: InputDescriptor) -> InputDescriptor: ...

    def stack(self, crops: list[np.ndarray]) -> torch.Tensor: ...


@dataclass
class SpatialPreparedImage:
    image: torch.Tensor
    support: torch.Tensor
    transform: CoordinateTransform


class SpatialTransform(Protocol):
    component_id: str

    def describe(self, source: InputDescriptor) -> InputDescriptor: ...

    def apply(
        self, image: torch.Tensor, origin_xy: tuple[int, int], frame_hw: tuple[int, int]
    ) -> SpatialPreparedImage: ...


class ModelBackend(Protocol):
    input_descriptor: InputDescriptor
    output_schema: str

    def predict(self, batch: PreparedBatch) -> Any: ...


class OutputDecoder(Protocol):
    input_schema: str

    def decode(self, output: Any, batch: PreparedBatch) -> list[EndpointResult]: ...


class QueryValidationError(ValueError):
    def __init__(self, code: str, message: str, status: str = "invalid_query"):
        super().__init__(message)
        self.code, self.status = code, status

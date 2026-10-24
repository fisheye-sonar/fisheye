"""Temporal selection is independent of image fetching and model execution."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True)
class ResolvedContext:
    frame_ids: tuple[int, ...]
    repeated: tuple[bool, ...]

    @property
    def context_frame_ids(self):
        return self.frame_ids

    @property
    def context_repeated(self):
        return self.repeated


class TrackTemporalSelector:
    component_id = "track_temporal.v1"

    def __init__(
        self,
        radius: int | None = None,
        offsets: Sequence[int] | None = None,
        basis: str = "track_positions",
        boundary: str = "repeat",
    ):
        if radius is not None and (
            isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 0
        ):
            raise ValueError("Temporal radius must be a nonnegative integer")
        if offsets is None:
            radius = 0 if radius is None else radius
            offsets = tuple(range(-radius, radius + 1))
        if (
            not offsets
            or any(isinstance(x, bool) or not isinstance(x, Integral) for x in offsets)
            or len(set(offsets)) != len(offsets)
            or 0 not in offsets
        ):
            raise ValueError(
                "Temporal offsets must be unique integers and contain zero"
            )
        if radius is not None and tuple(offsets) != tuple(range(-radius, radius + 1)):
            raise ValueError(
                "Temporal radius and offsets specify inconsistent contexts"
            )
        if basis not in ("track_positions", "source_frames"):
            raise ValueError("Temporal basis must be track_positions or source_frames")
        if boundary != "repeat":
            raise ValueError(
                "Only explicit repeat boundary handling is currently supported"
            )
        self.offsets, self.basis, self.boundary = (
            tuple(int(x) for x in offsets),
            basis,
            boundary,
        )

    def resolve(self, frame_ids: Sequence[int], center_index: int) -> ResolvedContext:
        if (
            not len(frame_ids)
            or isinstance(center_index, bool)
            or not isinstance(center_index, Integral)
            or not 0 <= center_index < len(frame_ids)
        ):
            raise ValueError(
                "Temporal center must identify an existing track observation"
            )
        if any(a >= b for a, b in zip(frame_ids, frame_ids[1:])):
            raise ValueError(
                "Temporal observations must be in strictly chronological order"
            )
        selected, repeated = [], []
        for offset in self.offsets:
            requested = (
                center_index + offset
                if self.basis == "track_positions"
                else frame_ids[center_index] + offset
            )
            lower, upper = (
                (0, len(frame_ids) - 1)
                if self.basis == "track_positions"
                else (frame_ids[0], frame_ids[-1])
            )
            position = min(max(requested, lower), upper)
            selected.append(
                int(
                    frame_ids[position] if self.basis == "track_positions" else position
                )
            )
            repeated.append(position != requested)
        return ResolvedContext(tuple(selected), tuple(repeated))

    def to_dict(self):
        return {
            "component_id": self.component_id,
            "offsets": list(self.offsets),
            "basis": self.basis,
            "boundary": self.boundary,
        }

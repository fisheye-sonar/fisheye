"""Native-pixel sonar side boundaries from the loader's calibrated beam angles."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SonarConeGeometry:
    angles_deg: tuple
    apex_xy_px: tuple
    inward_normals: tuple

    @classmethod
    def from_metadata(cls, metadata):
        """Use the same outer angular limits and metric grid as sonar warping.

        Unlike the legacy corner-fit helper, this uses actual outer beam edges,
        without rounding/clipping representative beam centers to image corners.
        """
        try:
            beam_data = metadata.beam_width_data
            left = np.asarray(beam_data["beam_left"], dtype=float)
            right = np.asarray(beam_data["beam_right"], dtype=float)
            scale = float(metadata.pixel_meter_size)
            origin = np.array(
                [metadata.x_meter_start, metadata.y_meter_start], dtype=float
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Sonar cone filtering requires calibrated beam and image metadata"
            ) from exc
        if (
            not left.size
            or not right.size
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
            or not np.isfinite(origin).all()
            or not np.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError(
                "Sonar cone filtering requires finite beam angles and positive pixel scale"
            )
        low, high = float(left.min()), float(right.max())
        if not -90 < low < high < 90:
            raise ValueError(
                "Sonar cone side angles must be ordered within (-90, 90) degrees"
            )
        a, b = np.deg2rad([low, high])
        # x_world = x_start + x_px*scale; y_world = y_start - y_px*scale.
        # Normals point into the cone, so negative distances identify exterior.
        return cls(
            (low, high),
            (float(-origin[0] / scale), float(origin[1] / scale)),
            (
                (float(np.cos(a)), float(np.sin(a))),
                (float(-np.cos(b)), float(-np.sin(b))),
            ),
        )

    def nearest_side_distance_px(self, points):
        """Smallest signed perpendicular distance over both endpoints and sides."""
        offsets = np.asarray(points, dtype=float) - self.apex_xy_px
        return float((offsets @ np.asarray(self.inward_normals).T).min())

    def to_dict(self):
        return {
            "source": "outer calibrated beam angles and loader metric pixel grid",
            "angles_deg": self.angles_deg,
            "apex_xy_px": self.apex_xy_px,
            "inward_normals": self.inward_normals,
            "distance": "signed perpendicular distance to the two sonar cone sides in native pixels",
        }

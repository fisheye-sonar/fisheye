"""Filter native-image endpoint observations, then aggregate each retained track."""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .config import TrackLengthConfig


@dataclass
class TrackLengths:
    prediction_filters: list
    tracks: list
    length_outputs: dict


def aggregate_track_lengths(results, frame_hw, config, *, cone, track_ids=()):
    """Count usable observations after filtering; interpolate percentiles linearly.

    Either endpoint at or within the margin of a cone side rejects the observation.
    Distances are perpendicular to the calibrated sides in native image pixels.
    Raw predictions are never mutated. Rejected tracks have no final length.
    """
    if not isinstance(config, TrackLengthConfig):
        config = TrackLengthConfig(**dict(config))
    height, width = frame_hw
    if height < 1 or width < 1:
        raise ValueError("Track length aggregation requires positive image dimensions")
    usable, observed, filters = defaultdict(list), Counter(), []
    identities = set(track_ids)
    seen = set()
    for result in results:
        track_id = result.track_id
        if track_id is not None:
            identities.add(track_id)
            observed[track_id] += 1
            identity = (track_id, result.frame_id)
            if identity in seen:
                raise ValueError("Duplicate endpoint observation within a track")
            seen.add(identity)
        reason, distance = None, None
        if track_id is None:
            reason = "missing_track_id"
        elif result.status != "ok":
            reason = result.status
        else:
            points = np.asarray(result.endpoints_xy_px, dtype=float)
            if points.shape != (2, 2) or not np.isfinite(points).all():
                reason = "invalid_endpoints"
            else:
                distance = cone.nearest_side_distance_px(points)
                if ((points < 0) | (points >= (width, height))).any():
                    reason = "endpoints_outside_frame"
                elif distance < 0:
                    reason = "endpoints_outside_sonar_cone"
                # A numerical tolerance keeps exactly-on-margin points rejected.
                elif distance <= config.edge_margin_px + 1e-9:
                    reason = "endpoint_near_cone_side"
                elif result.length_cm is None:
                    reason = "missing_physical_scale"
                elif (
                    not np.isfinite(result.length_cm)
                    or result.length_cm <= 0
                    or result.length_pixels is None
                    or not np.isfinite(result.length_pixels)
                    or result.length_pixels <= 0
                ):
                    reason = "invalid_length"
        filters.append(
            {
                "eligible": reason is None,
                "reason": reason,
                "nearest_cone_side_px": distance,
                "used_for_track_length": False,
            }
        )
        if reason is None:
            usable[track_id].append(result)

    tracks, outputs, accepted = [], {}, set()
    for track_id in sorted(identities):
        points = sorted(usable[track_id], key=lambda r: r.frame_id)
        summary = {
            "track_id": track_id,
            "prediction_count": observed[track_id],
            "usable_points": len(points),
            "discarded_predictions": observed[track_id] - len(points),
            "status": "discarded",
            "reason": "insufficient_usable_points",
            "percentile": config.percentile,
            "length_cm": None,
            "length_percentiles_cm": None,
            "length_pixels": None,
            "representative": None,
        }
        if len(points) >= config.min_track_points:
            lengths_cm = [p.length_cm for p in points]
            length_cm = float(
                np.percentile(lengths_cm, config.percentile, method="linear")
            )
            length_percentiles_cm = {
                str(percentile): float(value)
                for percentile, value in zip(
                    range(0, 101, 10),
                    np.percentile(lengths_cm, range(0, 101, 10), method="linear"),
                )
            }
            length_px = float(
                np.percentile(
                    [p.length_pixels for p in points],
                    config.percentile,
                    method="linear",
                )
            )
            # Preserve an actual observation for consumers requiring coordinates.
            # The representative's measured length may differ from interpolation.
            representative = min(points, key=lambda p: abs(p.length_cm - length_cm))
            summary.update(
                status="ok",
                reason=None,
                length_cm=length_cm,
                length_percentiles_cm=length_percentiles_cm,
                length_pixels=length_px,
                representative={
                    "query_id": representative.query_id,
                    "frame_id": representative.frame_id,
                    "endpoints_xy_px": representative.endpoints_xy_px,
                    "length_cm": representative.length_cm,
                    "length_pixels": representative.length_pixels,
                },
            )
            outputs[track_id] = {
                "filtered_lengths_cm": length_cm,
                "frame_id_closest_to_mean": representative.frame_id,
                "global_coords_px": representative.endpoints_xy_px,
                "length_method": "track_percentile",
                "length_percentile": config.percentile,
                "length_percentiles_cm": length_percentiles_cm,
                "length_observations": len(points),
                "representative_length_cm": representative.length_cm,
            }
            accepted.add(track_id)
        tracks.append(summary)
    for result, decision in zip(results, filters):
        decision["used_for_track_length"] = (
            decision["eligible"] and result.track_id in accepted
        )
    return TrackLengths(filters, tracks, outputs)

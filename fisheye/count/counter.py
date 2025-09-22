import structlog
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd

from fisheye.count.base import BaseCounter
from fisheye.enums import CountingMethod

from fisheye.utils import get_xy_from_r_theta
from fisheye.dataloaders.didson import pyARIS
from fisheye.configs.datasets import BEAM_WIDTH_DIR

logger = structlog.getLogger(__name__)


class LOICounter(BaseCounter):
    type = CountingMethod.LOI
    """Count fish by crossing the Line of Interest (LOI)."""

    def count(
        self,
        df: pd.DataFrame,
        line: float | tuple = 0.5,
        angle: float = None,
        metadata=None,
    ):
        """
        Count fish that cross a Line Of Interest (LOI).

        `line` can be:
        - float/int: vertical line x = line (matches current behavior; directed upward)
        - or ((x1, y1), (x2, y2)): a directed line from (x1,y1) -> (x2,y2)

        'left' means moved from the right side of the directed line to the left side.
        'right' means moved from the left side to the right side.
        `angle` is the angle of the LOI in degrees.
        `metadata` is the metadata of the ARIS file.

        if angle is specified the 'line' input is ignored
        """

        if angle is not None:
            if angle == 0.0:
                # take the vertical center line, this is done as often there isnt actually a beam at the center line
                line = ((0.5, 0), (0.5, 1))
            else:
                assert metadata is not None

                print(f"USING ANGLE NOT VERTICAL LOI count {angle=}")
                metadata.beam_width_data, _ = pyARIS.load_beam_width_data(
                    frame=metadata, beam_width_dir=BEAM_WIDTH_DIR
                )
                unwarped_r_min = metadata.y_meter_stop
                unwarped_r_max = metadata.y_meter_start

                near_point = get_xy_from_r_theta(unwarped_r_min, angle, metadata)
                far_point = get_xy_from_r_theta(unwarped_r_max, angle, metadata)
                line = (far_point, near_point)

        if False:
            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib.use("TkAgg")  # or "Qt5Agg"

            plt.ion()  # turn on interactive mode
            plt.scatter(
                df["x_center"],
                df["y_center"],
                c=df["frame"],  # color by time
                cmap="Greys",
            )
            plt.plot([0.5, 0.5], [0, 1], c="red", linestyle="--", label="center Line")
            plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], label="LOI")
            plt.colorbar(label="Time")  # optional colorbar
            # make y axis go from 1 to 0
            plt.gca().invert_yaxis()

            plt.show(block=True)  # force blocking window
        # --- normalize the LOI into two points and a direction vector ---
        if isinstance(line, (int, float)):
            x1, y1, x2, y2 = float(line), 0.0, float(line), 1.0  # vertical, pointing up
        else:
            try:
                (x1, y1), (x2, y2) = line
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            except Exception as e:
                raise ValueError("line must be a scalar x or ((x1,y1),(x2,y2))") from e

        dx, dy = x2 - x1, y2 - y1
        if abs(dx) + abs(dy) == 0:
            raise ValueError("Degenerate line: (x1,y1) == (x2,y2)")

        # signed side: >0 means point is to the LEFT of the arrow (x1,y1)->(x2,y2),
        # <0 means RIGHT, 0 means on the line.
        def signed_side(xs, ys):
            return dx * (ys - y1) - dy * (xs - x1)

        grouped_tracks = df.sort_values(by=["frame"], ascending=True).groupby("id")
        track_counts = defaultdict(lambda: {"left": 0, "right": 0})
        crossing_frames = {"left": [], "right": []}
        eps = 1e-9  # tolerance for "on the line"

        for track, track_df in grouped_tracks:
            xs = track_df["x_center"].to_numpy()
            ys = track_df["y_center"].to_numpy()
            frames = track_df["frame"].to_numpy()

            s_all = signed_side(xs, ys)

            # frame closest to LOI (min |signed distance|)
            closest_idx = int(np.argmin(np.abs(s_all)))
            x_center = float(xs[closest_idx])
            y_center = float(ys[closest_idx])
            width = float(track_df["width"].iloc[closest_idx])
            height = float(track_df["height"].iloc[closest_idx])
            closest_frame = int(frames[closest_idx])
            bbox = [x_center, y_center, width, height]

            s_first, s_last = float(s_all[0]), float(s_all[-1])

            # Crossed from RIGHT side to LEFT side -> "left"
            if s_first < -eps and s_last >= eps:
                track_counts[track]["left"] += 1
                crossing_frames["left"].append((track, closest_frame, bbox))
            # Crossed from LEFT side to RIGHT side -> "right"
            elif s_first > eps and s_last <= -eps:
                track_counts[track]["right"] += 1
                crossing_frames["right"].append((track, closest_frame, bbox))

        return self._calculate_absolute_counts(track_counts), crossing_frames

    @staticmethod
    def _calculate_absolute_counts(track_counts):
        """Calculate absolute counts for left and right."""
        absolute_left_counts = 0
        absolute_right_counts = 0

        for track, count in track_counts.items():
            net_counts = count["left"] - count["right"]
            if net_counts > 0:
                absolute_left_counts += 1
            else:
                absolute_right_counts += 1

        return absolute_left_counts, absolute_right_counts


class Count:
    """Main class to handle different counting methods."""

    def __init__(self, protocol: str = "loi"):
        """Initialize Count with a specific counting protocol."""
        self.protocol = protocol

        if self.protocol == CountingMethod.LOI:
            self.counter = LOICounter()
        else:
            raise ValueError(f"Protocol '{self.protocol}' is not supported.")

    def count(
        self,
        tracks: List[dict],
        line: float | tuple = 0.5,
        angle: float = None,
        metadata=None,
    ):
        """Count fish using the selected protocol.

        Args:
            tracks (dict): Dictionary containing tracking data for detected fish.
                Each entry should include bounding boxes in [x_center, y_center, width, height] format
                relative to the original image pixel space, with the following keys:
                    - 'frame': Frame index
                    - 'id': Unique track identifier
                    - 'x_center': X-coordinate of the bounding box center (in pixels)
                    - 'y_center': Y-coordinate of the bounding box center (in pixels)
                    - 'width': Bounding box width (in pixels)
                    - 'height': Bounding box height (in pixels)
                    - 'conf': Confidence score of the detection

        Returns:
            tuple: (absolute_left_count, absolute_right_count), crossing frames.
        """
        logger.info(f"initialized_counter", type=self.protocol)
        df = pd.DataFrame(tracks)
        if not df.empty:
            return self.counter.count(df, line, angle, metadata)

        logger.warning(f"No tracks available for counting.")
        # If no tracks present (empty dataframe) return 0 for both left and right counts
        return (0, 0), None

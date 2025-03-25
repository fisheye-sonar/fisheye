from collections import defaultdict

import numpy as np
import pandas as pd


class Count:
    def __init__(self):
        self.protocol = "LOI"

    def count(self, tracks: dict):
        if self.protocol == "LOI":
            mot_df = pd.DataFrame(tracks)
            mot_df["kp_x"] = mot_df["bb_left"] + mot_df["bb_width"] / 2
            mot_df["kp_y"] = mot_df["bb_top"] + mot_df["bb_height"] / 2
            return self.LOI_count(mot_df)

        else:
            raise ValueError(f"Protocol {self.protocol} not supported")

    def LOI_count(self, mot_tracks_df, line=0.5):
        """Count fish by line of interest (centerline)."""
        # Count the number of unique tracks that cross the line x = width/2
        unique_tracks = mot_tracks_df["id"].unique()
        track_counts = defaultdict(lambda: {"left": 0, "right": 0})

        for track in unique_tracks:
            track_df = mot_tracks_df[mot_tracks_df["id"] == track]
            # order by frame number
            track_df = track_df.sort_values(by="frame", ascending=True)
            # Check if the track crosses the line specified by line
            x_coords = track_df["kp_x"].values
            first_x = x_coords[0]
            last_x = x_coords[-1]

            # Judge by first and last x_coords
            if last_x <= line < first_x:
                track_counts[track]["left"] += 1

            elif last_x >= line > first_x:
                track_counts[track]["right"] += 1

            else:
                # Track does not cross the line
                pass

            # Find where the track crosses the line
            crossings = np.where((x_coords[:-1] - line) * (x_coords[1:] - line) <= 0)[0]

            # Handle edge cases
            if len(crossings) > 0:
                for i in crossings:
                    x1, x2 = x_coords[i], x_coords[i + 1]

                    # Skip if both points are on the line
                    if x1 == line and x2 == line:
                        continue

                    # Determine direction
                    if x1 < x2 or (x1 == line and x2 > line):
                        # left to right
                        track_counts[track]["right"] += 1
                    elif x1 > x2 or (x1 == line and x2 < line):
                        # right to left
                        track_counts[track]["left"] += 1

        # Calculate absolute counts
        absolute_left_counts = 0
        absolute_right_counts = 0

        for track, count in track_counts.items():
            net_counts = count["left"] - count["right"]
            if net_counts > 0:
                absolute_left_counts += 1
            else:
                absolute_right_counts += 1

        return absolute_left_counts, absolute_right_counts

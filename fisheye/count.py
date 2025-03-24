import asyncio
import glob
import io
import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import pandas as pd
import numpy as np
from collections import defaultdict

from sonar_cv.asset import AssetProcessor

__all__ = ["Count"]


def load_mot_file(file_path, img_width, img_height):
    df = pd.read_csv(file_path, delimiter=",", header=None)
    # df.columns = ['frame', 'id', 'x1', 'y1', 'width', 'height', 'conf', 'x', 'y', 'z']
    df.columns = ["frame", "id", "x1", "y1", "width", "height", "conf"]
    df["x"] = df["x1"].astype(float)
    df["y"] = df["y1"].astype(float)
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    df["width"] = df["width"].astype(float)
    df["height"] = df["height"].astype(float)
    df["conf"] = df["conf"].astype(float)
    # Scale normalized coordinates to image pixel dimensions
    # df['x'] = (df['x1'] * img_width).astype(int)
    # df['y'] = (df['y1'] * img_height).astype(int)
    # df['frame'] = df['frame'].astype(int)
    # df['id'] = df['id'].astype(int)
    # df['width'] = (df['width'] * img_width).astype(int)
    # df['height'] = (df['height'] * img_height).astype(int)
    # df['conf'] = df['conf'].astype(float)

    # Convert bbox to centroids (keypoints) where x and y are the center of the bbox and columns in the df
    df["kp_x"] = df["x"] + df["width"] / 2
    df["kp_y"] = df["y"] + df["height"] / 2

    return df


class Count:
    def __init__(self):
        self.protocol = "LOI"

    def count(self, mot_tracks_file):
        if self.protocol == "LOI":
            raw_file = Path(mot_tracks_file).stem + ".aris"
            # fp = os.path.join(Path(mot_tracks_file).parent, raw_file)
            fp = f"/Volumes/Memorex USB/2025_01_CA-Caltech_meeting[Van_Duzen_River]_Data/Videos/{raw_file}"

            imgs = asyncio.run(AssetProcessor().generate_frames(fp))

            image = Image.open(io.BytesIO(imgs[0]["content"]))
            # get height and width
            w, h = image.size

            # line_x = w/2
            line_x = 0.5

            return self.LOI_count(mot_tracks_file, line_x, w, h)
        else:
            raise ValueError(f"Protocol {self.protocol} not supported")

    def LOI_count(self, mot_tracks_file, line, w, h):
        """Count fish based on the line of interest (center line)."""
        mot_tracks_df = load_mot_file(mot_tracks_file, w, h)
        # Count the number of unique tracks that cross the line x = width/2
        unique_tracks = mot_tracks_df["id"].unique()
        track_counts = defaultdict(lambda: {"left": 0, "right": 0})
        for track in unique_tracks:
            track_df = mot_tracks_df[mot_tracks_df["id"] == track]
            # order by frame number
            track_df = track_df.sort_values(by="frame", ascending=True)

            # Check if the track crosses the line specified by line
            x_coords = track_df["kp_x"].values

            # save first and last x_coords
            first_x = x_coords[0]
            last_x = x_coords[-1]

            # Judge by first and last x_coords
            if last_x <= line and first_x > line:
                track_counts[track]["left"] += 1
            elif last_x >= line and first_x < line:
                track_counts[track]["right"] += 1
            else:
                pass

            # Find where the track crosses the line
            crossings = np.where((x_coords[:-1] - line) * (x_coords[1:] - line) <= 0)[0]
            print(f"Fish ID {track} crosses the line at frames: {crossings}")

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

        print(
            f"Absolute left counts: {absolute_left_counts}, Absolute right counts: {absolute_right_counts}"
        )
        return absolute_left_counts, absolute_right_counts

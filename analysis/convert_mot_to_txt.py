import pandas as pd
import json
from fisheye.utils import get_unwarped_distance, get_theta
import os


class DictNamespace:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(value))
            else:
                setattr(self, key, value)


def detect_centerline_crossings(
    center_line, frame_ids, x_center, y_center, upstream_direction
):
    """
    Detect when x coordinates cross a centerline

    Args:
        frame_ids: Series of frame IDs corresponding to x coordinates
        center_line: The centerline x coordinate to check for crossings
        x_center: Series of x coordinates
        y_center: Series of y coordinates
        upstream_direction: "left" or "right"
    Returns:
        List of tuples (frame_id, direction) where direction is 'left_to_right' or 'right_to_left'
    """
    # MAH 2025-06-30 15:14:27 bumping the center line so that we never have the x cordinate bang on the center line
    center_line = center_line - 0.1
    crossings = []

    for i in range(1, len(x_center)):
        prev_x = x_center.iloc[i - 1]
        curr_x = x_center.iloc[i]
        curr_y = y_center.iloc[i]
        frame_id = frame_ids.iloc[i]

        if prev_x == (center_line + 0.1):
            print(
                f"\033[33mWARNING: a point lands bang on the center line {prev_x=} == center_line={center_line+0.1} this shouldnt be a problem as we have nudged the centerline by 0.1\033[0m"
            )
        if upstream_direction == "left":
            left_to_right = "down"
            right_to_left = "up"
        else:
            left_to_right = "up"
            right_to_left = "down"
        # Check if crossing occurred
        if prev_x < center_line and curr_x >= center_line:
            # Left to right crossing
            crossings.append((frame_id, left_to_right, curr_x, curr_y))
        elif prev_x > center_line and curr_x <= center_line:
            # Right to left crossing
            crossings.append((frame_id, right_to_left, curr_x, curr_y))

    return crossings


def mot_to_txt(
    mot_path,
    info_path,
    start_frame,
    upstream_direction,
    output_path=None,
    verbose=False,
):
    """
    Convert an MOT bounding box file to a text file with centerline crossings
    Args:
        mot_path: Path to the MOT file
        info_path: Path to the info file
        start_frame: Start frame of the MOT file, needed as the MOT file is reindexed for each clip
        output_path: Path to the output file
        verbose: Whether to print verbose output
    Returns:
        None
    """
    with open(info_path, "r") as f:
        info = json.load(f)

    xdim = info["xdim"]
    ydim = info["ydim"]
    half_xdim = xdim / 2

    info["half_xdim"] = half_xdim
    # Read the MOT file
    df = pd.read_csv(mot_path)
    df.columns = [
        "frame_id",
        "track_id",
        "x",
        "y",
        "w",
        "h",
        "confidence",
        "_x",
        "_y",
        "_z",
    ]

    # Calculate center points
    df["x_center"] = df["x"] + df["w"] / 2
    df["y_center"] = df["y"] + df["h"] / 2
    # Get the center line from the info file
    center_line = info["half_xdim"]

    num_crossings = 0
    all_crossings = []

    # Get each fish's velocity
    unique_fish_ids = df["track_id"].unique()
    if verbose:
        print(f"Unique fish IDs: {unique_fish_ids}")
    for track_id in unique_fish_ids:
        track_df = df[df["track_id"] == track_id].sort_values("frame_id")
        if verbose:
            print(f"Fish ID: {track_id}")
            print(f"Number of frames: {len(track_df)}")

        # Detect centerline crossings
        crossings = detect_centerline_crossings(
            center_line,
            track_df["frame_id"],
            track_df["x_center"],
            track_df["y_center"],
            upstream_direction,
        )
        if verbose:
            print(f"{crossings=}")
        if crossings:
            num_crossings += 1
            if verbose:
                print(f"  Centerline crossings for fish {track_id}:")
            for frame_id, direction, x_center, y_center in crossings:
                if verbose:
                    print(f"    Frame {frame_id}: {direction}")
                # Store crossing data for output file
                # get_unwarped_distance
                info_dict = DictNamespace(info)

                row = {}
                row["metadata"] = info_dict
                row["bbox"] = [
                    x_center / info_dict.xdim,
                    y_center / info_dict.ydim,
                    0,
                    0,
                ]
                #  crossing["frame_id"]
                distance = get_unwarped_distance(row)
                theta = get_theta(row)
                all_crossings.append(
                    {
                        "track_id": track_id,
                        "frame_id": frame_id,
                        "direction": direction,
                        "distance": distance,
                        "theta": theta,
                    }
                )
        else:
            if verbose:
                print(f"  No centerline crossings for fish {track_id}")

        if verbose:
            print(
                f"  X coordinate range: {track_df['x_center'].min():.2f} to {track_df['x_center'].max():.2f}"
            )
            print(f"  Centerline: {center_line:.2f}")

    if verbose:
        print(f"Number of crossings: {num_crossings} {all_crossings}")

    # Save crossings to text file in the same format as manual marking
    with open(output_path, "w") as f:
        # Write header
        f.write("*** Centerline Crossings ***\n\n")

        headers = [
            "File",
            "Total",
            "Frame#",
            "Dir",
            "R (m)",
            "Theta",
            "L(cm)",
            "dR(cm)",
            "L/dR",
            "Aspect",
            "Time",
            "Date",
            "Latitude",
            "Longitude",
            "Pan",
            "Tilt",
            "Roll",
            "Species",
            "Motion",
            "Q",
            "N",
            "Comment",
        ]

        col_width = 10
        header_line = "".join(f"{h:<{col_width}}" for h in headers)
        separator_line = "-" * len(header_line)
        f.write(header_line + "\n")
        f.write(separator_line + "\n")

        # Write each crossing
        for i, crossing in enumerate(all_crossings, 1):
            # Format: File Total Frame# Dir R(m) Theta L(cm) dR(cm) L/dR Aspect Time Date Lat Long Pan Tilt Roll Species Motion Q N Comment
            line = f"{1:<10} {i:<10} {crossing['frame_id']+start_frame:<10} {crossing['direction']:<10} {crossing['distance']:<10.2f} {crossing['theta']: <10.1f} {0.0:<10.1f} {0.0:<10.1f} {0.0:<10.1f} {0.0:<10.1f} {'00:00:00':<10} {'2018-07-02':<10} {'N 00 d  0.00000 m':<18} {'E 000 d  0.00000 m':<18} {0.0:<10.1f} {0.0:<10.1f} {0.0:<10.1f} {'Unknown':<10} {'Running':<12} {5:<3} {1:<3} {'Centerlinecrossing'}trackid{crossing['track_id']}\n"
            f.write(line)

        print(f"Crossings saved to: {output_path}")


if __name__ == "__main__":
    # /home/mahobley/Data/CFC22/restructured_dataset/annotations/elwha/Elwha_2018_OM_ARIS_2018_07_10_2018-07-10_040000_6750_7201/ MISSING AN ANNOTATED FISH
    # Elwha_2018_OM_ARIS_2018_07_27_2018-07-27_030000_897_1348_r_m we are detecting an extra fish here
    mot_annotations_dir = "PATHTODATASET/annotations/elwha/Elwha_2018_OM_ARIS_2018_07_10_2018-07-10_040000_6750_7201/"
    info_dir = (
        "PATHTODATASET/info/elwha/Elwha_2018_OM_ARIS_2018_07_10_2018-07-10_040000/"
    )
    output_dir = "/home/mahobley/Code/fisheye/"

    info_fn = "Elwha_2018_OM_ARIS_2018_07_10_2018-07-10_040000.json"
    mot_fn = "gt.txt"
    output_fn = f"{mot_annotations_dir.split('/')[-1]}_crossings_generated_from_annotations_test.txt"

    info_path = os.path.join(info_dir, info_fn)
    mot_path = os.path.join(mot_annotations_dir, mot_fn)
    output_path = os.path.join(output_dir, output_fn)
    start_frame = int(mot_path.split("_")[-3])

    x = mot_to_txt(mot_path, info_path, start_frame, output_path, verbose=True)

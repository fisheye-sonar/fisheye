import os
import sys
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


from collections import defaultdict
import copy


def extract_fields(data, field, direction=None):
    if direction:
        return [d[field] for d in data if d["direction"] == direction]
    return [d[field] for d in data]


def remove_opposite_pairs(data):
    # Step 1: Group by track_id
    track_groups = defaultdict(list)
    for entry in data:
        track_groups[entry["track_id"]].append(entry)

    # Step 2: For each track_id group
    result = []
    for track_id, entries in track_groups.items():
        if track_id == -1:
            continue
        # Work on a copy so we can safely remove items
        group = copy.deepcopy(entries)
        while True:
            # Find all up/down pairs
            pairs = [
                (i, j, abs(group[i]["frame_id"] - group[j]["frame_id"]))
                for i in range(len(group))
                for j in range(i + 1, len(group))
                if group[i]["direction"] != group[j]["direction"]
            ]
            if not pairs:
                break
            # Find the closest pair
            i, j, _ = min(pairs, key=lambda x: x[2])
            # Remove both entries
            for idx in sorted([i, j], reverse=True):
                del group[idx]
        if len(group) > 1:
            print(f"MAYBE ERROR CANCELLED TO NOT 1 {group=}")
        result.extend(group)
    return result


def match_points_by_axis(
    gt_frame_ids, pred_frame_ids, gt_r_m, pred_r_m, max_frame_diff=3, max_r_diff=0.2
):
    """
    Match predicted points to ground truth points using independent axis thresholds.

    Returns:
    - matches: list of (gt_index, pred_index)
    - unmatched_gt: list of gt indices without match
    - unmatched_pred: list of pred indices without match
    """

    potential_matches = []

    # Build a list of all valid pairs with their distance (for sorting)
    for gt_idx, (gt_frame, gt_r) in enumerate(zip(gt_frame_ids, gt_r_m)):
        for pred_idx, (pred_frame, pred_r) in enumerate(zip(pred_frame_ids, pred_r_m)):
            frame_diff = abs(gt_frame - pred_frame)
            r_diff = abs(gt_r - pred_r)
            if frame_diff <= max_frame_diff and r_diff <= max_r_diff:
                # Euclidean distance (or any simple score combining the two axes)
                dist = (frame_diff**2 + r_diff**2) ** 0.5
                potential_matches.append((dist, gt_idx, pred_idx))

    # Sort by distance (closest first)
    potential_matches.sort()

    matches = []
    matched_gt = set()
    matched_pred = set()

    for dist, gt_idx, pred_idx in potential_matches:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matches.append((gt_idx, pred_idx))
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)

    unmatched_gt = [i for i in range(len(gt_frame_ids)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_frame_ids)) if i not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def parse_file(file_path):
    """
    Parse a text file with the specified format and extract frame_id, direction, and R(m)
    """
    data = []
    up_count = 0
    down_count = 0

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Skip header lines (lines starting with *** or containing dashes)
        data_lines = []

        for line in lines:
            line = line.strip()

            if (
                line
                and not line.startswith("***")
                and not line.startswith("---")
                and not line.startswith("File")
                and not line == "\n"
            ):
                data_lines.append(line)

        # Parse data lines
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 5:  # Ensure we have enough columns
                try:

                    total_frame = int(parts[1])  # Total column
                    frame_id = int(parts[2])  # Frame# column
                    direction = parts[3].lower()  # Dir column (convert to lowercase)
                    r_m = float(parts[4])  # R (m) column
                    theta = float(parts[5])  # Theta column
                    comments = parts[-1]
                    if "Centerlinecrossingtrackid" in comments:
                        track_id = int(comments.split("Centerlinecrossingtrackid")[1])
                    else:
                        track_id = total_frame

                    data.append(
                        {
                            "frame_id": frame_id,
                            "direction": direction,
                            "r_m": r_m,
                            "theta": theta,
                            "track_id": track_id,
                        }
                    )

                    # Count directions
                    if direction == "up":
                        up_count += 1
                    elif direction == "down":
                        down_count += 1

                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line}, {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, 0, 0
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, 0, 0

    return data, up_count, down_count


def compare_2_txt_files(
    file_A,
    file_B,
    output_filepath,
    plot=False,
    save_json_per_clip=False,
    save_plots_per_clip=False,
    max_frame_diff=10,
    max_r_diff=0.2,
    remove_multiple_crossings_per_track=False,
    verbose=False,
):
    # Define the two files to compare
    name_A = "gt"
    name_B = "pred"
    # print(f"=== Comparing {gt_name} and {pred_name} ===\n")

    # Parse both files
    if verbose:
        print(f"Parsing {name_A}: {file_A}")
    data_A, up_A, down_A = parse_file(file_A)
    if verbose:
        print(f"Parsing {name_B}: {file_B}")
    data_B, up_B, down_B = parse_file(file_B)

    if data_A is None:
        print(f"Error: Could not parse {file_A}")
        return
    if data_B is None:
        print(f"Error: Could not parse {file_A}")
        return

    # Summary comparison
    if verbose:
        print(f"=== Summary Comparison ===")
        print(
            f"{name_A}: {up_A} up, {down_A} down, {len(data_A)} total, {up_A-down_A} net"
        )
        print(
            f"{name_B}: {up_B} up, {down_B} down, {len(data_B)} total, {up_B-down_B} net"
        )

    total_upstream_gt = sum(1 for d in data_A if d["direction"] == "up")
    total_downstream_gt = sum(1 for d in data_A if d["direction"] == "down")
    total_upstream_pred = sum(1 for d in data_B if d["direction"] == "up")
    total_downstream_pred = sum(1 for d in data_B if d["direction"] == "down")

    # if a single track has multiple crossings, we only count the net movement (cancelling out paired up crossings)
    data_A_net_movement_per_track = remove_opposite_pairs(data_A)
    data_B_net_movement_per_track = remove_opposite_pairs(data_B)

    upstream_net_movement_by_track_A = sum(
        1 for d in data_A_net_movement_per_track if d["direction"] == "up"
    )
    downstream_net_movement_by_track_A = sum(
        1 for d in data_A_net_movement_per_track if d["direction"] == "down"
    )
    upstream_net_movement_by_track_B = sum(
        1 for d in data_B_net_movement_per_track if d["direction"] == "up"
    )
    downstream_net_movement_by_track_B = sum(
        1 for d in data_B_net_movement_per_track if d["direction"] == "down"
    )

    if verbose:
        print(f"{upstream_net_movement_by_track_A=}")
        print(f"{downstream_net_movement_by_track_A=}")
        print(f"{upstream_net_movement_by_track_B=}")
        print(f"{downstream_net_movement_by_track_B=}")

    # total_crossings_error = (
    #     abs(total_upstream_gt - total_upstream_pred)
    #     + abs(total_downstream_gt - total_downstream_pred)
    # ) / (total_upstream_gt + total_downstream_gt + 1e-10)
    # total_count_error = (
    #     abs(upstream_net_movement_by_track_A - upstream_net_movement_by_track_B)
    #     + abs(downstream_net_movement_by_track_A - downstream_net_movement_by_track_B)
    # ) / (upstream_net_movement_by_track_A + downstream_net_movement_by_track_A + 1e-10)
    # net_count_error = (
    #     (total_upstream_pred - total_downstream_pred)
    #     - (total_upstream_gt - total_downstream_gt)
    # ) / (total_upstream_gt - total_downstream_gt + 1e-10)

    # if verbose:
    # print(f"Total Crossings Error: {total_crossings_error*100:.2f}%")
    # print(f"Total Count Error: {total_count_error*100:.2f}%")
    # print(f"Net Count Error:   {net_count_error*100:.2f}%")

    info_dict = {
                    f"data_summary_{name_A}": {
                        "total_upstream_crossings": total_upstream_gt,
                        "total_downstream_crossings": total_downstream_gt,
                        "upstream_net_movement_by_track": upstream_net_movement_by_track_A,
                        "downstream_net_movement_by_track": downstream_net_movement_by_track_A,
                        "total_crossings": total_upstream_gt + total_downstream_gt,
                        "net_crossings": total_upstream_gt - total_downstream_gt,
                        "net_counts": upstream_net_movement_by_track_A
                        - downstream_net_movement_by_track_A,
                    },
                    f"data_summary_{name_B}": {
                        "total_upstream_crossings": total_upstream_pred,
                        "total_downstream_crossings": total_downstream_pred,
                        "upstream_net_movement_by_track": upstream_net_movement_by_track_B,
                        "downstream_net_movement_by_track": downstream_net_movement_by_track_B,
                        "total_crossings": total_upstream_pred + total_downstream_pred,
                        "net_crossings": total_upstream_pred - total_downstream_pred,
                        "net_counts": upstream_net_movement_by_track_B
                        - downstream_net_movement_by_track_B,
                    },
                    # "total_crossings_error": total_crossings_error,
                    # "total_count_error": total_count_error,
                    # "net_count_error": net_count_error,
                }
    # save the data_summary_gt and data_summary_pred to a json file
    if save_json_per_clip and output_filepath is not None:
        with open(output_filepath + ".json", "w") as f:
            json.dump(
                info_dict,
                f,
                indent=4,
            )

    if plot:
        if remove_multiple_crossings_per_track:
            data_A = data_A_net_movement_per_track
            data_B = data_B_net_movement_per_track
        for data_type in ["r_m", "theta"]:
            plt.figure(figsize=(16, 9))  # 16 inches wide, 9 inches tall

            track_ids_A = extract_fields(data_A, "track_id")
            track_ids_B = extract_fields(data_B, "track_id")

            up_frame_ids_A = extract_fields(data_A, "frame_id", "up")
            down_frame_ids_A = extract_fields(data_A, "frame_id", "down")
            up_frame_ids_B = extract_fields(data_B, "frame_id", "up")
            down_frame_ids_B = extract_fields(data_B, "frame_id", "down")

            up_dat_A = extract_fields(data_A, data_type, "up")
            down_dat_A = extract_fields(data_A, data_type, "down")
            up_dat_B = extract_fields(data_B, data_type, "up")
            down_dat_B = extract_fields(data_B, data_type, "down")

            if verbose:
                print(f"Matching up points")
            matches_up, unmatched_gt_up, unmatched_pred_up = match_points_by_axis(
                up_frame_ids_A,
                up_frame_ids_B,
                up_dat_A,
                up_dat_B,
                max_frame_diff=max_frame_diff,
                max_r_diff=max_r_diff,
            )

            if verbose:
                print(f"Matching down points")
            matches_down, unmatched_gt_down, unmatched_pred_down = match_points_by_axis(
                down_frame_ids_A,
                down_frame_ids_B,
                down_dat_A,
                down_dat_B,
                max_frame_diff=max_frame_diff,
                max_r_diff=max_r_diff,
            )

            handles = []
            labels = []

            for unmatched_gt_up_idx in unmatched_gt_up:
                h = plt.scatter(
                    [
                        up_frame_ids_A[unmatched_gt_up_idx],
                    ],
                    [up_dat_A[unmatched_gt_up_idx]],
                    color="yellow",
                    s=100,
                    alpha=0.75,
                )
                handles.append(h)
                labels.append("Unmatched GT")
            for unmatched_gt_down_idx in unmatched_gt_down:
                plt.scatter(
                    [
                        down_frame_ids_A[unmatched_gt_down_idx],
                    ],
                    [down_dat_A[unmatched_gt_down_idx]],
                    color="yellow",
                    s=100,
                    alpha=0.75,
                )
            for unmatched_pred_up_idx in unmatched_pred_up:
                h = plt.scatter(
                    [
                        up_frame_ids_B[unmatched_pred_up_idx],
                    ],
                    [up_dat_B[unmatched_pred_up_idx]],
                    color="orange",
                    s=100,
                    alpha=0.5,
                )
                handles.append(h)
                labels.append("Unmatched pred")
            for unmatched_pred_down_idx in unmatched_pred_down:
                plt.scatter(
                    [
                        down_frame_ids_B[unmatched_pred_down_idx],
                    ],
                    [down_dat_B[unmatched_pred_down_idx]],
                    color="orange",
                    s=100,
                    alpha=0.5,
                )

            for gt_track_id in set(track_ids_A):
                gt_fids = [
                    d["frame_id"] for d in data_A if d["track_id"] == gt_track_id
                ]
                dat_A_track_id = [
                    d[data_type] for d in data_A if d["track_id"] == gt_track_id
                ]
                if len(gt_fids) > 1:
                    h = plt.plot(
                        gt_fids,
                        dat_A_track_id,
                        alpha=0.5,
                        color="blue",
                    )
                    handles.append(h[0])
                    labels.append("GT same track")

            for pred_track_id in set(track_ids_B):
                pred_fids = [
                    d["frame_id"] for d in data_B if d["track_id"] == pred_track_id
                ]
                dat_B_track_id = [d[data_type] for d in data_B if d["track_id"]]
                if len(pred_fids) > 1:
                    h = plt.plot(
                        pred_fids,
                        dat_B_track_id,
                        alpha=0.5,
                        color="red",
                    )
                    handles.append(h[0])
                    labels.append("Pred same track")

            h = plt.scatter(
                up_frame_ids_A,
                up_dat_A,
                alpha=0.5,
                color="blue",
                marker="^",
            )
            handles.append(h)
            labels.append("GT 'up'")
            h = plt.scatter(
                down_frame_ids_A,
                down_dat_A,
                alpha=0.5,
                color="blue",
                marker="v",
            )
            handles.append(h)
            labels.append("GT 'down'")
            h = plt.scatter(
                up_frame_ids_B,
                up_dat_B,
                alpha=0.5,
                color="red",
                marker="^",
            )
            handles.append(h)
            labels.append("Pred 'up'")
            h = plt.scatter(
                down_frame_ids_B,
                down_dat_B,
                alpha=0.5,
                color="red",
                marker="v",
            )
            handles.append(h)
            labels.append("Pred 'down'")
            unique = dict()
            for h, l in zip(handles, labels):
                if l not in unique:
                    unique[l] = h
            plt.legend(unique.values(), unique.keys())
            if len(up_dat_A + down_dat_A + up_dat_B + down_dat_B) > 0:
                vmin = min(up_dat_A + down_dat_A + up_dat_B + down_dat_B) - 1
                vmax = max(up_dat_A + down_dat_A + up_dat_B + down_dat_B) + 1
            else:
                vmin = 0
                vmax = 1
            plt.ylim(vmin - 1, vmax + 1)
            if (
                len(
                    up_frame_ids_A
                    + down_frame_ids_A
                    + up_frame_ids_B
                    + down_frame_ids_B
                )
                > 0
            ):
                xmin = min(
                    up_frame_ids_A
                    + down_frame_ids_A
                    + up_frame_ids_B
                    + down_frame_ids_B
                )
                xmax = max(
                    up_frame_ids_A
                    + down_frame_ids_A
                    + up_frame_ids_B
                    + down_frame_ids_B
                )
            else:
                xmin = 0
                xmax = 1

            plt.xlim(int(xmin - 10), int(xmax + 10))

            plt.xlabel("Frame ID")
            if data_type == "r_m":
                plt.ylabel("Distance R (m)")
            elif data_type == "theta":
                plt.ylabel("Angle (deg)")
            plt.title(
                f"{data_type} vs Frame ID\n{output_filepath.split('/')[-2:]}\n"
                f"{name_A}: counts: {upstream_net_movement_by_track_A} up, {downstream_net_movement_by_track_A} down, crossings: {total_upstream_gt} up, {total_downstream_gt} down, {total_upstream_gt+total_downstream_gt} total crossings, {upstream_net_movement_by_track_A-downstream_net_movement_by_track_A} net\n"
                f"{name_B}: counts: {upstream_net_movement_by_track_B} up, {downstream_net_movement_by_track_B} down,crossings: {total_upstream_pred} up, {total_downstream_pred} down, {total_upstream_pred+total_downstream_pred} total crossings, {upstream_net_movement_by_track_B-downstream_net_movement_by_track_B} net\n"
                # f"Total Count Error: {total_count_error*100:.2f}% "
                # f"Total Crossings Error: {total_crossings_error*100:.2f}%, "
                # f"Net Count Error: {net_count_error*100:.2f}%"
            )
            # plt.legend()
            if save_plots_per_clip:
                plt.savefig(output_filepath + f"_{data_type}.png")
                plt.close()
            else:
                plt.show()

    return info_dict


if __name__ == "__main__":
    frame_range = "3600_3900"
    pred_file = f"/home/mahobley/Code/fisheye/analysis/RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_{frame_range}_crossings_generated_from_annotations.txt"
    gt_file = f"/home/mahobley/Code/fisheye/results/FCe_RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_ID_{frame_range}_cropped.txt"
    
    output_dir = "/home/mahobley/Code/fisheye/analysis/outputs"
    output_file = (
        f"FCe_RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_ID_{frame_range}"
    )
    output_filepath = os.path.join(output_dir, output_file)

    compare_2_txt_files(
        gt_file,
        pred_file,
        output_filepath,
        plot=True,
        save_plots_per_clip=True,
        save_json_per_clip=True,
        max_frame_diff=10,
        max_r_diff=0.2,
        remove_multiple_crossings_per_track=False,
    )

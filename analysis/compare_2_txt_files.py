import os
import sys
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


from collections import defaultdict
import copy


def remove_opposite_pairs(data_pred):
    # Step 1: Group by track_id
    track_groups = defaultdict(list)
    for entry in data_pred:
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


# # Example usage
# data_pred = [
#     {"frame_id": 3601, "direction": "up", "r_m": 4.42, "theta": 0.0, "track_id": 628},
#     {"frame_id": 3605, "direction": "up", "r_m": 3.86, "theta": 0.0, "track_id": 631},
#     {"frame_id": 3609, "direction": "up", "r_m": 3.35, "theta": 0.0, "track_id": 633},
#     {"frame_id": 3616, "direction": "up", "r_m": 4.2, "theta": 0.0, "track_id": 632},
#     {"frame_id": 3621, "direction": "up", "r_m": 2.91, "theta": 0.0, "track_id": 635},
#     {"frame_id": 3627, "direction": "up", "r_m": 3.46, "theta": 0.0, "track_id": 637},
#     {"frame_id": 3628, "direction": "up", "r_m": 5.21, "theta": 0.0, "track_id": 634},
#     {"frame_id": 3642, "direction": "up", "r_m": 3.44, "theta": 0.0, "track_id": 639},
#     {"frame_id": 3647, "direction": "up", "r_m": 5.55, "theta": 0.0, "track_id": 636},
# ]
# filtered = remove_opposite_pairs(data_pred)
# print(filtered)


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
    import heapq

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


# def match_points_by_axis(
#     gt_frame_ids, pred_frame_ids, gt_r_m, pred_r_m, max_frame_diff=3, max_r_diff=0.2
# ):
#     """
#     Match predicted points to ground truth points using independent axis thresholds.

#     Parameters:
#     - gt_points: list of (frame_id, R)
#     - pred_points: list of (frame_id, R)
#     - max_frame_diff: maximum allowed difference in frame ID (x-axis)
#     - max_r_diff: maximum allowed difference in R value (y-axis)

#     Returns:
#     - matches: list of (gt_index, pred_index)
#     - unmatched_gt: list of gt indices without match
#     - unmatched_pred: list of pred indices without match
#     """
#     matches = []
#     unmatched_gt = []
#     matched_pred_indices = set()

#     for gt_idx, (gt_frame, gt_r) in enumerate(zip(gt_frame_ids, gt_r_m)):
#         found_match = False
#         for pred_idx, (pred_frame, pred_r) in enumerate(zip(pred_frame_ids, pred_r_m)):
#             if pred_idx in matched_pred_indices:
#                 continue
#             if (
#                 abs(gt_frame - pred_frame) <= max_frame_diff
#                 and abs(gt_r - pred_r) <= max_r_diff
#             ):
#                 matches.append((gt_idx, pred_idx))
#                 matched_pred_indices.add(pred_idx)
#                 found_match = True
#                 break
#         if not found_match:
#             unmatched_gt.append(gt_idx)

#     unmatched_pred = [
#         i for i in range(len(pred_frame_ids)) if i not in matched_pred_indices
#     ]

#     return matches, unmatched_gt, unmatched_pred


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
            # print(f"{parts[-1]=}")
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
        return None, 0, 0, 0
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, 0, 0, 0

    data_summary = {
        "up": up_count,
        "down": down_count,
        "total": len(data),
        "net": up_count - down_count,
    }

    return data, up_count, down_count, data_summary


def compare_2_txt_files(
    gt_file,
    pred_file,
    output_filepath,
    plot=False,
    save=False,
    max_frame_diff=10,
    max_r_diff=0.2,
    remove_multiple_tracks=False,
    verbose=False,
):
    # Define the two files to compare

    gt_name = "GT  "
    pred_name = "Pred"

    # print(f"=== Comparing {gt_name} and {pred_name} ===\n")

    # Parse both files
    if verbose:
        print(f"Parsing {gt_name}: {gt_file}")
    data_gt, up_gt, down_gt, data_summary_gt = parse_file(gt_file)

    if verbose:
        print(f"Parsing {pred_name}: {pred_file}")
    data_pred, up_pred, down_pred, data_summary_pred = parse_file(pred_file)
    if data_gt is None or data_pred is None:
        print("Error: Could not parse one or both files")
        return

    # Summary comparison
    if verbose:
        print(f"=== Summary Comparison ===")
        print(
            f"{gt_name}: {up_gt} up, {down_gt} down, {len(data_gt)} total, {up_gt-down_gt} net"
        )
        print(
            f"{pred_name}: {up_pred} up, {down_pred} down, {len(data_pred)} total, {up_pred-down_pred} net"
        )

    total_upstream_gt = sum(1 for d in data_gt if d["direction"] == "up")
    total_downstream_gt = sum(1 for d in data_gt if d["direction"] == "down")
    total_upstream_pred = sum(1 for d in data_pred if d["direction"] == "up")
    total_downstream_pred = sum(1 for d in data_pred if d["direction"] == "down")

    data_gt_cancelled = remove_opposite_pairs(data_gt)
    data_pred_cancelled = remove_opposite_pairs(data_pred)

    cancel_upstream_gt = sum(1 for d in data_gt_cancelled if d["direction"] == "up")
    cancel_downstream_gt = sum(1 for d in data_gt_cancelled if d["direction"] == "down")
    cancel_upstream_pred = sum(1 for d in data_pred_cancelled if d["direction"] == "up")
    cancel_downstream_pred = sum(
        1 for d in data_pred_cancelled if d["direction"] == "down"
    )

    if verbose:
        print(f"{cancel_upstream_gt=}")
        print(f"{cancel_downstream_gt=}")
        print(f"{cancel_upstream_pred=}")
        print(f"{cancel_downstream_pred=}")

    total_crossings_error = (
        abs(total_upstream_gt - total_upstream_pred)
        + abs(total_downstream_gt - total_downstream_pred)
    ) / (total_upstream_gt + total_downstream_gt + 1e-10)
    total_count_error = (
        abs(cancel_upstream_gt - cancel_upstream_pred)
        + abs(cancel_downstream_gt - cancel_downstream_pred)
    ) / (cancel_upstream_gt + cancel_downstream_gt + 1e-10)
    net_count_error = (
        (total_upstream_pred - total_downstream_pred)
        - (total_upstream_gt - total_downstream_gt)
    ) / (total_upstream_gt - total_downstream_gt + 1e-10)

    if verbose:
        print(f"Total Crossings Error: {total_crossings_error*100:.2f}%")
        print(f"Total Count Error: {total_count_error*100:.2f}%")
        print(f"Net Count Error:   {net_count_error*100:.2f}%")

    # save the data_summary_gt and data_summary_pred to a json file
    with open(output_filepath + ".json", "w") as f:
        json.dump(
            {
                "data_summary_gt": {
                    "total_upstream_crossings": total_upstream_gt,
                    "total_downstream_crossings": total_downstream_gt,
                    "cancel_upstream_crossings": cancel_upstream_gt,
                    "cancel_downstream_crossings": cancel_downstream_gt,
                    "total_crossings": total_upstream_gt + total_downstream_gt,
                    "net_crossings": total_upstream_gt - total_downstream_gt,
                    "net_counts": cancel_upstream_gt - cancel_downstream_gt,
                },
                "data_summary_pred": {
                    "total_upstream_crossings": total_upstream_pred,
                    "total_downstream_crossings": total_downstream_pred,
                    "cancel_upstream_crossings": cancel_upstream_pred,
                    "cancel_downstream_crossings": cancel_downstream_pred,
                    "total_crossings": total_upstream_pred + total_downstream_pred,
                    "net_crossings": total_upstream_pred - total_downstream_pred,
                    "net_counts": cancel_upstream_pred - cancel_downstream_pred,
                },
                "total_crossings_error": total_crossings_error,
                "total_count_error": total_count_error,
                "net_count_error": net_count_error,
            },
            f,
            indent=4,
        )

    if remove_multiple_tracks:
        data_gt = data_gt_cancelled
        data_pred = data_pred_cancelled

    if plot:
        for data_type in ["r_m", "theta"]:
            plt.figure(figsize=(16, 9))  # 16 inches wide, 9 inches tall

            gt_frame_ids = [d["frame_id"] for d in data_gt]
            pred_frame_ids = [d["frame_id"] for d in data_pred]
            gt_dat = [d[data_type] for d in data_gt]
            pred_dat = [d[data_type] for d in data_pred]
            gt_track_ids = [d["track_id"] for d in data_gt]
            pred_track_ids = [d["track_id"] for d in data_pred]

            gt_up_frame_ids = [d["frame_id"] for d in data_gt if d["direction"] == "up"]
            gt_down_frame_ids = [
                d["frame_id"] for d in data_gt if d["direction"] == "down"
            ]
            pred_up_frame_ids = [
                d["frame_id"] for d in data_pred if d["direction"] == "up"
            ]
            pred_down_frame_ids = [
                d["frame_id"] for d in data_pred if d["direction"] == "down"
            ]

            gt_up_dat = [d[data_type] for d in data_gt if d["direction"] == "up"]
            gt_down_dat = [d[data_type] for d in data_gt if d["direction"] == "down"]
            pred_up_dat = [d[data_type] for d in data_pred if d["direction"] == "up"]
            pred_down_dat = [
                d[data_type] for d in data_pred if d["direction"] == "down"
            ]

            if verbose:
                print(f"Matching up points")
            matches_up, unmatched_gt_up, unmatched_pred_up = match_points_by_axis(
                gt_up_frame_ids,
                pred_up_frame_ids,
                gt_up_dat,
                pred_up_dat,
                max_frame_diff=max_frame_diff,
                max_r_diff=max_r_diff,
            )

            if verbose:
                print(f"Matching down points")
            matches_down, unmatched_gt_down, unmatched_pred_down = match_points_by_axis(
                gt_down_frame_ids,
                pred_down_frame_ids,
                gt_down_dat,
                pred_down_dat,
                max_frame_diff=max_frame_diff,
                max_r_diff=max_r_diff,
            )

            # for match in matches_up:
            #     plt.plot(
            #         [gt_up_frame_ids[match[0]], pred_up_frame_ids[match[1]]],
            #         [gt_up_r_m[match[0]], pred_up_r_m[match[1]]],
            #         color="green",
            #     )

            # for match in matches_down:
            #     plt.plot(
            #         [gt_down_frame_ids[match[0]], pred_down_frame_ids[match[1]]],
            #         [gt_down_r_m[match[0]], pred_down_r_m[match[1]]],
            #         color="orange",
            #     )
            handles = []
            labels = []

            for unmatched_gt_up_idx in unmatched_gt_up:
                h = plt.scatter(
                    [
                        gt_up_frame_ids[unmatched_gt_up_idx],
                    ],
                    [gt_up_dat[unmatched_gt_up_idx]],
                    color="yellow",
                    s=100,
                    alpha=0.75,
                )
                handles.append(h)
                labels.append("Unmatched GT")
            for unmatched_gt_down_idx in unmatched_gt_down:
                plt.scatter(
                    [
                        gt_down_frame_ids[unmatched_gt_down_idx],
                    ],
                    [gt_down_dat[unmatched_gt_down_idx]],
                    color="yellow",
                    s=100,
                    alpha=0.75,
                )
            for unmatched_pred_up_idx in unmatched_pred_up:
                h = plt.scatter(
                    [
                        pred_up_frame_ids[unmatched_pred_up_idx],
                    ],
                    [pred_up_dat[unmatched_pred_up_idx]],
                    color="orange",
                    s=100,
                    alpha=0.5,
                )
                handles.append(h)
                labels.append("Unmatched pred")
            for unmatched_pred_down_idx in unmatched_pred_down:
                plt.scatter(
                    [
                        pred_down_frame_ids[unmatched_pred_down_idx],
                    ],
                    [pred_down_dat[unmatched_pred_down_idx]],
                    color="orange",
                    s=100,
                    alpha=0.5,
                )

            for gt_track_id in set(gt_track_ids):
                gt_fids = [
                    d["frame_id"] for d in data_gt if d["track_id"] == gt_track_id
                ]
                gt_dat = [d[data_type] for d in data_gt if d["track_id"] == gt_track_id]
                if len(gt_fids) > 1:
                    h = plt.plot(
                        gt_fids,
                        gt_dat,
                        alpha=0.5,
                        color="blue",
                    )
                    handles.append(h[0])
                    labels.append("GT same track")

            for pred_track_id in set(pred_track_ids):
                pred_fids = [
                    d["frame_id"] for d in data_pred if d["track_id"] == pred_track_id
                ]
                pred_dat = [d[data_type] for d in data_pred if d["track_id"]]
                if len(pred_fids) > 1:
                    h = plt.plot(
                        pred_fids,
                        pred_dat,
                        alpha=0.5,
                        color="red",
                    )
                    handles.append(h[0])
                    labels.append("Pred same track")

            h = plt.scatter(
                gt_up_frame_ids,
                gt_up_dat,
                alpha=0.5,
                color="blue",
                marker="^",
            )
            handles.append(h)
            labels.append("GT 'up'")
            h = plt.scatter(
                gt_down_frame_ids,
                gt_down_dat,
                alpha=0.5,
                color="blue",
                marker="v",
            )
            handles.append(h)
            labels.append("GT 'down'")
            h = plt.scatter(
                pred_up_frame_ids,
                pred_up_dat,
                alpha=0.5,
                color="red",
                marker="^",
            )
            handles.append(h)
            labels.append("Pred 'up'")
            h = plt.scatter(
                pred_down_frame_ids,
                pred_down_dat,
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
            if len(gt_up_dat + gt_down_dat + pred_up_dat + pred_down_dat) > 0:
                vmin = min(gt_up_dat + gt_down_dat + pred_up_dat + pred_down_dat) - 1
                vmax = max(gt_up_dat + gt_down_dat + pred_up_dat + pred_down_dat) + 1
            else:
                vmin = 0
                vmax = 1
            plt.ylim(vmin - 1, vmax + 1)
            if (
                len(
                    gt_up_frame_ids
                    + gt_down_frame_ids
                    + pred_up_frame_ids
                    + pred_down_frame_ids
                )
                > 0
            ):
                xmin = min(
                    gt_up_frame_ids
                    + gt_down_frame_ids
                    + pred_up_frame_ids
                    + pred_down_frame_ids
                )
                xmax = max(
                    gt_up_frame_ids
                    + gt_down_frame_ids
                    + pred_up_frame_ids
                    + pred_down_frame_ids
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
                f"{gt_name}: counts: {cancel_upstream_gt} up, {cancel_downstream_gt} down, crossings: {total_upstream_gt} up, {total_downstream_gt} down, {total_upstream_gt+total_downstream_gt} total crossings, {cancel_upstream_gt-cancel_downstream_gt} net\n"
                f"{pred_name}: counts: {cancel_upstream_pred} up, {cancel_downstream_pred} down,crossings: {total_upstream_pred} up, {total_downstream_pred} down, {total_upstream_pred+total_downstream_pred} total crossings, {cancel_upstream_pred-cancel_downstream_pred} net\n"
                f"Total Count Error: {total_count_error*100:.2f}% "
                f"Total Crossings Error: {total_crossings_error*100:.2f}%, "
                f"Net Count Error: {net_count_error*100:.2f}%"
            )
            # plt.legend()
            if save:
                plt.savefig(output_filepath + f"_{data_type}.png")
                plt.close()
            else:
                plt.show()


if __name__ == "__main__":
    # frame_range = "900_1200"
    frame_range = "3600_3900"
    # frame_range = "3900_4200"
    # frame_range = "6000_6300"
    gt_file = f"/home/mahobley/Code/fisheye/analysis/RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_{frame_range}_crossings_generated_from_annotations.txt"
    pred_file = f"/home/mahobley/Code/fisheye/results/FCe_RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_ID_{frame_range}.txt"
    pred_file = f"/home/mahobley/Code/fisheye/analysis/RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_{frame_range}_crossings_generated_from_annotations.txt"
    pred_file = f"/home/mahobley/Code/fisheye/analysis/generated_results/nushagak/FCe_RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_ID__900_1200_cropped.txt"
    pred_file = f"/home/mahobley/Code/fisheye/analysis/generated_results/kenai-rightbank/FCe_2018-05-26-JD146_RightFar_Stratum2_Set1_RO_2018-05-26_151004_ID__1560_1760_cropped.txt"
    gt_file = f"/home/mahobley/Code/fisheye/results/FCe_RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_ID_{frame_range}.txt"
    gt_file = f"/home/mahobley/Code/fisheye/analysis/gt_files/nushagak/FCe_RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_ID__900_1200_cropped.txt"
    gt_file = f"/home/mahobley/Code/fisheye/analysis/gt_files/kenai-rightbank/FCe_2018-05-26-JD146_RightFar_Stratum2_Set1_RO_2018-05-26_151004_ID__1560_1760_cropped.txt"
    # gt_file = "/home/mahobley/Code/fisheye/results/FCe_2018-05-26-JD146_LeftFar_Stratum1_Set1_LO_2018-05-26_080004_ID_.txt"
    # pred_file = "/home/mahobley/Code/fisheye/results/FCe_2018-05-26-JD146_LeftFar_Stratum1_Set1_LO_2018-05-26_080004_ID_285_885.txt"
    output_dir = "/home/mahobley/Code/fisheye/analysis/outputs"
    output_file = (
        f"FCe_RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_ID_{frame_range}"
    )
    # output_file = (
    #     f"FCe_2018-05-26-JD146_LeftFar_Stratum1_Set1_LO_2018-05-26_080004_ID_285_885"
    # )

    output_filepath = os.path.join(output_dir, output_file)
    compare_2_txt_files(
        gt_file,
        pred_file,
        output_filepath,
        plot=True,
        save=True,
        max_frame_diff=10,
        max_r_diff=0.2,
        remove_multiple_tracks=False,
    )

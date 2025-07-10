import os
import json
import matplotlib.pyplot as plt
import numpy as np


def read_jsons_for_location(json_parent_dir, location):
    json_dir = os.path.join(json_parent_dir, location)

    dict_list = []
    # Lists to store extracted values

    # Load data from JSON files
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir, filename), "r") as f:
                data = json.load(f)
                dict_list.append(data)

    return dict_list


def dict_list_to_datas(dict_list):
    gt_net_counts = []
    pred_net_counts = []
    gt_upstream_crossings = []
    gt_downstream_crossings = []
    pred_upstream_crossings = []
    pred_downstream_crossings = []
    gt_net_crossings = []
    pred_net_crossings = []
    for data in dict_list:
        gt = data["data_summary_gt"]["net_counts"]
        pred = data["data_summary_pred"]["net_counts"]
        upstream_net_movement_by_track_gt = data["data_summary_gt"][
            "upstream_net_movement_by_track"
        ]
        downstream_net_movement_by_track_gt = data["data_summary_gt"][
            "downstream_net_movement_by_track"
        ]
        upstream_net_movement_by_track_pred = data["data_summary_pred"][
            "upstream_net_movement_by_track"
        ]
        downstream_net_movement_by_track_pred = data["data_summary_pred"][
            "downstream_net_movement_by_track"
        ]

        net_crossings_gt = data["data_summary_gt"]["net_crossings"]
        net_crossings_pred = data["data_summary_pred"]["net_crossings"]

        gt_net_counts.append(gt)
        pred_net_counts.append(pred)
        gt_upstream_crossings.append(upstream_net_movement_by_track_gt)
        gt_downstream_crossings.append(downstream_net_movement_by_track_gt)
        pred_upstream_crossings.append(upstream_net_movement_by_track_pred)
        pred_downstream_crossings.append(downstream_net_movement_by_track_pred)
        gt_net_crossings.append(net_crossings_gt)
        pred_net_crossings.append(net_crossings_pred)
    return (
        gt_net_counts,
        pred_net_counts,
        gt_upstream_crossings,
        gt_downstream_crossings,
        pred_upstream_crossings,
        pred_downstream_crossings,
        gt_net_crossings,
        pred_net_crossings,
    )


def analyse_location(dict_list, location, print_for_latex, plot, save_json_path=""):
    location_name_str = location[0].upper() + location[1:]

    (
        gt_net_counts,
        pred_net_counts,
        gt_upstream_crossings,
        gt_downstream_crossings,
        pred_upstream_crossings,
        pred_downstream_crossings,
        gt_net_crossings,
        pred_net_crossings,
    ) = dict_list_to_datas(dict_list)

    net_count_errors = list(np.array(pred_net_counts) - np.array(gt_net_counts))

    if print_for_latex:
        overpredicted = (
            np.array(pred_upstream_crossings) - np.array(pred_downstream_crossings)
        ) > (np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))
        underpredicted = (
            np.array(pred_upstream_crossings) - np.array(pred_downstream_crossings)
        ) < (np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))
        num_upstream_clips = sum(
            np.array(gt_upstream_crossings) > np.array(gt_downstream_crossings)
        )
        num_downstream_clips = sum(
            np.array(gt_upstream_crossings) < np.array(gt_downstream_crossings)
        )
        # print for latex table
        print(
            f" & {len(gt_upstream_crossings)} & {sum(gt_upstream_crossings)} & {sum(gt_downstream_crossings)} & \\textbf{{{sum(gt_upstream_crossings) - sum(gt_downstream_crossings)}}} & {sum(pred_upstream_crossings)} & {sum(pred_downstream_crossings)} & \\textbf{{{sum(pred_upstream_crossings) - sum(pred_downstream_crossings)}}}  & {num_upstream_clips} & {num_downstream_clips}  & {sum(overpredicted)} & {sum(underpredicted)}\\\\"
        )

    nMAE = sum(
        np.abs(np.array(pred_upstream_crossings) - np.array(gt_upstream_crossings))
        + np.abs(
            np.array(pred_downstream_crossings) - np.array(gt_downstream_crossings)
        )
    ) / sum(gt_upstream_crossings + gt_downstream_crossings)

    nMNE = sum(
        (np.array(pred_upstream_crossings) - np.array(pred_downstream_crossings))
        - (np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))
    ) / sum(np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))
    Net_absolute_metric = sum(
        np.abs(
            (np.array(pred_upstream_crossings) - np.array(pred_downstream_crossings))
            - (np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))
        )
    ) / sum(np.abs(np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings)))

    video_dominant_motion_sign = np.where(
        np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings) >= 0, 1, -1
    )
    nMANE = sum(
        video_dominant_motion_sign
        * (
            (np.array(pred_upstream_crossings) - np.array(pred_downstream_crossings))
            - (np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))
        )
    ) / sum(np.abs(np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings)))

    metrics = {
        "nMAE": nMAE,
        "nMNE": nMNE,
        "nMANE": nMANE,
    }

    metrics_str = (
        f"nMAE: {nMAE*100:.2f}%, nMNE: {nMNE*100:.2f}%, nMANE: {nMANE*100:.2f}%"
    )

    # print(f"'{location}': {{")
    # for metric, value in metrics.items():
    #     print(f"'{metric}': {value*100:.2f},")
    # print(f"}},")
    if save_json_path:
        with open(os.path.join(save_json_path, f"{location}.json"), "w") as f:
            json.dump(metrics, f)

    if plot:

        # net_count_errors = [x if x < 1000 else -200 for x in net_count_errors]

        zipped = zip(
            gt_net_counts,
            pred_net_counts,
            net_count_errors,
        )
        zipped_sorted = sorted(zipped, key=lambda x: x[0])
        # Filter out entries where list3 (x[2]) is 0
        # zipped_sorted = [item for item in zipped_sorted if item[0] != 0]
        (
            gt_net_counts,
            pred_net_counts,
            net_count_errors,
        ) = zip(*zipped_sorted)

        # -------- Plot 1: GT vs Pred net counts --------

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 5)
        ax.grid(True)

        for i in range(len(gt_net_counts)):
            ax.plot(
                [i, i],
                [gt_net_counts[i], pred_net_counts[i]],
                color="k",
                alpha=0.5,
                zorder=1,
            )
        ax.scatter(
            range(len(gt_net_counts)),
            gt_net_counts,
            color="blue",
            label="GT Net Count",
            # alpha=0.5,
            zorder=2,
            # s=10,
        )
        ax.scatter(
            range(len(pred_net_counts)),
            pred_net_counts,
            color="red",
            label="Pred Net Count",
            # alpha=0.5,
            marker="x",
            s=10,
            zorder=2,
        )

        ax.scatter(
            range(len(net_count_errors)),
            net_count_errors,
            color="orange",
            marker="+",
            s=5,
            label="Pred Net Error",
            alpha=0.5,
            zorder=3,
        )

        ax.set_title(f"{location.capitalize()} GT vs Pred Net Counts\n{metrics_str}")
        ax.set_xlabel("Clip Index (ordered by GT Net Count)")
        ax.set_ylabel("Net Counts")
        ax.legend()
        plt.tight_layout()
        ax.set_axisbelow(True)

        plt.savefig(f"analysis/figures/{location}_gt_vs_pred_net_counts.png", dpi=300)
        plt.close("all")

        # -------- Plot: Histogram of GT Net Counts --------
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 5)
        ax.grid(True)
        bin_edges = np.arange(min(net_count_errors), max(net_count_errors) + 1, 1)
        ax.hist(net_count_errors, bins=bin_edges, color="skyblue", edgecolor="black")
        # log plot
        ax.set_title(
            f"{location.capitalize()} Histogram of Net Count Errors\n{metrics_str}"
        )
        ax.set_xlabel("Net Count Error")
        ax.set_ylabel("Frequency")
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(
            f"analysis/figures/{location}_histogram_of_net_count_errors.png",
            dpi=300,
        )
        plt.close("all")

        # -------- Plot: Histogram of Net Count Error Percentages --------
        net_count_errors_percentages = [
            x / gt for x, gt in zip(net_count_errors, gt_net_counts) if gt != 0
        ]
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 5)
        # bin_edges = np.arange(
        #     min(net_count_errors_percentages), max(net_count_errors_percentages) + 1, 1
        # )

        ax.hist(
            net_count_errors_percentages,
            bins=50,
            color="skyblue",
            edgecolor="black",
        )
        # log plot
        ax.set_title(
            f"{location.capitalize()} Histogram of Net Count Error Percentages\n{metrics_str}"
        )
        ax.set_xlabel("Net Count Error Percentage")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(
            f"analysis/figures/{location}_histogram_of_net_count_error_percentages.png",
            dpi=300,
        )
        plt.close("all")
        # plt.show()

    return metrics


if __name__ == "__main__":

    print_for_latex = False
    plot = True

    # Directory with JSON files
    json_parent_dir = "/home/mahobley/Code/fisheye/analysis/outputs"
    # json_dir = "/home/mahobley/Code/fisheye/analysis/outputs/kenai-rightbank"
    # json_dir = "/home/mahobley/Code/fisheye/analysis/outputs/nushagak"

    locations = [
        "nushagak",
        "elwha",
        "kenai-rightbank",
        "kenai-val",
        "kenai-train",
        "kenai-channel",
    ]

    for location in locations:
        dict_list = read_jsons_for_location(json_parent_dir, location)

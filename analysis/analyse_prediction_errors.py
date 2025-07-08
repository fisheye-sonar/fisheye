import os
import json
import matplotlib.pyplot as plt
import numpy as np

# flip_sign = -1
# flip_sign = 1
# Directory with JSON files
json_parent_dir = "/home/mahobley/Code/fisheye/analysis/outputs"
# json_dir = "/home/mahobley/Code/fisheye/analysis/outputs/kenai-rightbank"
# json_dir = "/home/mahobley/Code/fisheye/analysis/outputs/nushagak"

locations_dirs = {
    "nushagak": 1,
    "elwha": -1,
    "kenai-rightbank": 1,
    "kenai-val": -1,
    "kenai-train": -1,
    "kenai-channel": -1,
}

# location, flip_sign = list(locations_dirs.items())[3]
for location, flip_sign in locations_dirs.items():
    json_dir = os.path.join(json_parent_dir, location)

    # Lists to store extracted values
    gt_net_counts = []
    pred_net_counts = []
    net_count_errors = []
    total_count_errors = []
    gt_upstream_crossings = []
    gt_downstream_crossings = []
    pred_upstream_crossings = []
    pred_downstream_crossings = []
    gt_net_crossings = []
    pred_net_crossings = []

    error_3 = []

    # Load data from JSON files
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir, filename), "r") as f:
                data = json.load(f)
                gt = data["data_summary_gt"]["net_counts"]
                pred = data["data_summary_pred"]["net_counts"]
                error = data["net_count_error"]
                error_2 = data["total_count_error"]

                gt_net_counts.append(flip_sign * gt)
                pred_net_counts.append(flip_sign * pred)
                net_count_errors.append(error)
                total_count_errors.append(error_2)
                error_3.append(
                    flip_sign
                    * (
                        (
                            data["data_summary_pred"]["cancel_upstream_crossings"]
                            - data["data_summary_pred"]["cancel_downstream_crossings"]
                        )
                        - (
                            data["data_summary_gt"]["cancel_upstream_crossings"]
                            - data["data_summary_gt"]["cancel_downstream_crossings"]
                        )
                    )
                )

                if flip_sign == -1:
                    gt_upstream_crossings.append(
                        data["data_summary_gt"]["cancel_downstream_crossings"]
                    )
                    gt_downstream_crossings.append(
                        data["data_summary_gt"]["cancel_upstream_crossings"]
                    )
                    pred_upstream_crossings.append(
                        data["data_summary_pred"]["cancel_downstream_crossings"]
                    )
                    pred_downstream_crossings.append(
                        data["data_summary_pred"]["cancel_upstream_crossings"]
                    )
                else:
                    gt_upstream_crossings.append(
                        data["data_summary_gt"]["cancel_upstream_crossings"]
                    )
                    gt_downstream_crossings.append(
                        data["data_summary_gt"]["cancel_downstream_crossings"]
                    )
                    pred_upstream_crossings.append(
                        data["data_summary_pred"]["cancel_upstream_crossings"]
                    )
                    pred_downstream_crossings.append(
                        data["data_summary_pred"]["cancel_downstream_crossings"]
                    )
                gt_net_crossings.append(data["data_summary_gt"]["net_crossings"])
                pred_net_crossings.append(data["data_summary_pred"]["net_crossings"])

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
    if False:
        # print for latex table
        print(
            f" & {len(gt_upstream_crossings)} & {sum(gt_upstream_crossings)} & {sum(gt_downstream_crossings)} & \\textbf{{{sum(gt_upstream_crossings) - sum(gt_downstream_crossings)}}} & {sum(pred_upstream_crossings)} & {sum(pred_downstream_crossings)} & \\textbf{{{sum(pred_upstream_crossings) - sum(pred_downstream_crossings)}}}  & {num_upstream_clips} & {num_downstream_clips}  & {sum(overpredicted)} & {sum(underpredicted)}\\\\"
        )

    # print(f" & {num_overpredicted=} & {num_underpredicted=} ")

    # print(
    #     f"{underpredicted * (np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))}"
    # )

    Original_metric = sum(
        np.abs(np.array(pred_upstream_crossings) - np.array(gt_upstream_crossings))
        + np.abs(
            np.array(pred_downstream_crossings) - np.array(gt_downstream_crossings)
        )
    ) / sum(gt_upstream_crossings + gt_downstream_crossings)

    # print(f"{gt_upstream_crossings=} {gt_downstream_crossings=}")
    # print(
    #     f"{sum(np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))=}"
    # )
    # print(
    #     f"{sum(np.abs(np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings)))=}"
    # )
    Net_metric = sum(
        (np.array(pred_upstream_crossings) - np.array(pred_downstream_crossings))
        - (np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))
    ) / sum(np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))
    Net_absolute_metric = sum(
        np.abs(
            (np.array(pred_upstream_crossings) - np.array(pred_downstream_crossings))
            - (np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))
        )
    ) / sum(np.abs(np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings)))
    sign = np.where(
        np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings) >= 0, 1, -1
    )
    error_sign = sign * (
        (np.array(pred_upstream_crossings) - np.array(pred_downstream_crossings))
        - (np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))
    )

    Net_absolute_metric_2 = sum(
        sign
        * (
            (np.array(pred_upstream_crossings) - np.array(pred_downstream_crossings))
            - (np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings))
        )
    ) / sum(np.abs(np.array(gt_upstream_crossings) - np.array(gt_downstream_crossings)))

    print(f"'{location}': {{")
    print(f"'Original_metric': {Original_metric*100:.2f},")
    print(f"'Net_metric': {Net_metric*100:.2f},")
    print(f"'Net_absolute_metric': {Net_absolute_metric*100:.2f},")
    print(f"'Net_absolute_metric_2': {Net_absolute_metric_2*100:.2f},")
    print(f"}},")

    net_count_errors = [x if x < 1000 else -200 for x in net_count_errors]

    zipped = zip(
        gt_net_counts,
        pred_net_counts,
        net_count_errors,
        total_count_errors,
        error_3,
        error_sign,
        sign,
    )
    zipped_sorted = sorted(zipped, key=lambda x: x[0])
    # Filter out entries where list3 (x[2]) is 0
    # zipped_sorted = [item for item in zipped_sorted if item[0] != 0]
    (
        gt_net_counts,
        pred_net_counts,
        net_count_errors,
        total_count_errors,
        error_3,
        error_sign,
        sign,
    ) = zip(*zipped_sorted)
    mean_gt_net_count = np.mean(gt_net_counts)
    mean_pred_net_count = np.mean(pred_net_counts)
    mean_abs_net_count_error = np.mean([abs(x) for x in net_count_errors])

    mean_net_count_error = np.mean(net_count_errors)
    mean_total_count_error = np.mean(total_count_errors)
    # print(f"{net_count_errors=} {mean_net_count_error=}")
    # print(f"{pred_net_counts_flip_negative=} {mean_flipped_net_count_error=}")

    # print(f"'{location}': {{")
    # print(f"'mean_gt_net_count': {mean_gt_net_count:.2f},")
    # print(f"'mean_pred_net_count': {mean_pred_net_count:.2f},")
    # print(f"'mean_abs_net_count_error_percent': {mean_abs_net_count_error*100:.2f},")
    # print(f"'mean_net_count_error_percent': {mean_net_count_error*100:.2f},")
    # print(f"'mean_total_count_error_percent': {mean_total_count_error*100:.2f},")
    # print(f"}},")
    # continue
    # -------- Plot 1: GT vs Pred net counts --------
    plot = True
    if plot:
        plt.figure(figsize=(10, 5))
        plt.grid(True)

        for i in range(len(gt_net_counts)):
            plt.plot(
                [i, i],
                [gt_net_counts[i], pred_net_counts[i]],
                color="k",
                alpha=0.5,
                zorder=1,
            )
        plt.scatter(
            range(len(gt_net_counts)),
            gt_net_counts,
            color="blue",
            label="GT Net Count",
            # alpha=0.5,
            zorder=2,
            # s=10,
        )
        plt.scatter(
            range(len(pred_net_counts)),
            pred_net_counts,
            color="red",
            label="Pred Net Count",
            # alpha=0.5,
            marker="x",
            s=10,
            zorder=2,
        )
        # plt.scatter(
        #     range(len(net_count_errors)),
        #     net_count_errors,
        #     color="orange",
        #     marker="x",
        #     s=10,
        #     label="Pred Net Error Perc",
        #     alpha=1,
        # )
        plt.scatter(
            range(len(error_3)),
            error_3,
            color="orange",
            marker="o",
            s=5,
            label="Pred Net Error",
            alpha=0.5,
            zorder=3,
        )
        # plt.scatter(
        #     range(len(error_sign)),
        #     error_sign,
        #     color="c",
        #     marker="o",
        #     s=1,
        #     label="Pred Net Error 2",
        #     alpha=1,
        #     zorder=3,
        # )

        plt.title(f"{location} GT vs Pred Net Counts")
        plt.xlabel("Clip Index (ordered by GT Net Count)")
        plt.ylabel("Net Counts")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"analysis/figures/{location}_gt_vs_pred_net_counts.png", dpi=300)

        # -------- Plot 2: Net Count Error vs GT Count --------
        plt.figure(figsize=(8, 5))
        plt.scatter(gt_net_counts, net_count_errors, color="green", alpha=0.25)
        plt.title(f"{location} Net Count Error vs GT Net Count")
        plt.xlabel("GT Net Count")
        plt.ylabel("Net Count Error")
        plt.grid(True)
        plt.tight_layout()

        # -------- Plot 3: Histogram of GT Net Counts --------
        plt.figure(figsize=(8, 5))
        plt.hist(net_count_errors, bins=20, color="skyblue", edgecolor="black")
        # log plot
        plt.title(f"{location} Histogram of Net Count Errors")
        plt.xlabel("Net Count Error")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        # print(f"{net_count_errors=}")

        # -------- Plot 3: Histogram of GT Net Counts --------
        plt.figure(figsize=(8, 5))
        plt.hist(total_count_errors, bins=20, color="skyblue", edgecolor="black")
        # log plot
        plt.title(f"{location} Histogram of Total Count Errors")
        plt.xlabel("Total Count Error")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()

        # -------- Plot 2: Net Count Error vs GT Count --------
        plt.figure(figsize=(8, 5))
        min_error = min(min(total_count_errors), min(net_count_errors))
        max_error = max(max(total_count_errors), max(net_count_errors))

        plt.scatter(
            total_count_errors,
            [abs(x) for x in net_count_errors],
            color="green",
            alpha=0.5,
        )
        # get the limits of the plot
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        plt.plot([min_error, max_error], [min_error, max_error], color="k", alpha=0.1)
        # reset the limits of the plot
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.title(f"{location} Total Count Error vs Net Count Error")
        plt.xlabel("Total Count Error")
        plt.ylabel("|Net Count Error|")
        plt.grid(True)
        plt.tight_layout()
        plt.close("all")
        # plt.show()

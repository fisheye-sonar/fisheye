import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_across_locations(
    results, output_path, metrics=["nMAE", "nMNE", "nMANE"]
):
    # check all results have all the metrics
    for loc in results.keys():
        for metric in metrics:
            if metric not in results[loc].keys():
                raise ValueError(f"Metric {metric} not found for location {loc}")

    # Metrics to plot
    # metrics = ["nMAE", "nMNE", "nMANE"]

    # Set up
    locations = list(results.keys())
    x = np.arange(len(locations))  # label locations
    x_labels = [loc.capitalize() for loc in locations]
    width = 0.2  # width of the bars

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    ax.axhline(
        y=10,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="10% threshold",
        alpha=0.4,
    )
    ax.axhline(y=-10, color="red", linestyle="--", linewidth=1.5, alpha=0.4)
    ax.axhline(y=0, color="k", linewidth=1.5, alpha=1)

    # Plot each metric as a bar set
    for i, metric in enumerate(metrics):
        values = [results[loc][metric] * 100 for loc in locations]
        # ax.bar(x + i * width - width, values, width, label=metric, color=colors[i])
        offset = i * width - width  # adjust bar position
        ax.bar(x + offset, values, width, label=metric, color=colors[i])

        # Draw top lines for absolute values
        for xi, val in zip(x + offset, values):
            if val < 0:
                ax.hlines(
                    abs(val),
                    xi - width / 2,
                    xi + width / 2,
                    color=colors[i],
                    linewidth=2,
                    linestyles="dotted",
                )

    # Labeling
    ax.set_xlabel("Location")
    ax.set_ylabel("Error Percent")
    ax.set_title("Error Percent by Location and Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    min_y = (
        int(min([results[loc][metric] for loc in locations for metric in metrics])) - 1
    )
    max_y = (
        int(max([results[loc][metric] for loc in locations for metric in metrics])) + 1
    )
    min_y = min(min_y, -10)
    max_y = max(max_y, 10)
    ax.set_yticks(np.arange(min_y - 1, max_y + 1, 1))

    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    output_path = "analysis/figures/error_percent_by_location.png"

    results = {
        "nushagak": {
            "nMAE": 0.00,
            "nMNE": 1.00,
            "nMANE": 1.00,
        },
        "elwha": {
            "nMAE": 1.00,
            "nMNE": 1.00,
            "nMANE": 1.00,
        },
    }

    plot_metrics_across_locations(results, output_path)

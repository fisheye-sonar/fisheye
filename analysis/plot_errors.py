import matplotlib.pyplot as plt
import numpy as np


results = {
    "nushagak": {
        "Original_metric": 2.48,
        "Net_metric": -1.13,
        "Net_absolute_metric": 2.42,
        "Net_absolute_metric_2": -1.13,
    },
    "elwha": {
        "Original_metric": 14.03,
        "Net_metric": -1.07,
        "Net_absolute_metric": 14.11,
        "Net_absolute_metric_2": -9.09,
    },
    "kenai-rightbank": {
        "Original_metric": 2.75,
        "Net_metric": 0.15,
        "Net_absolute_metric": 2.70,
        "Net_absolute_metric_2": -1.09,
    },
    "kenai-val": {
        "Original_metric": 5.46,
        "Net_metric": -4.79,
        "Net_absolute_metric": 4.62,
        "Net_absolute_metric_2": -4.62,
    },
    "kenai-train": {
        "Original_metric": 9.41,
        "Net_metric": 0.16,
        "Net_absolute_metric": 9.22,
        "Net_absolute_metric_2": -3.07,
    },
    "kenai-channel": {
        "Original_metric": 10.00,
        "Net_metric": 5.74,
        "Net_absolute_metric": 11.81,
        "Net_absolute_metric_2": 0.69,
    },
}
# Metrics to plot
metrics = {
    "Original_metric": "nMAE",
    # "Net_absolute_metric": "Absolute Net Count Error",
    "Net_metric": "nMNE",
    "Net_absolute_metric_2": "nMANE",
}

# Set up
locations = list(results.keys())
x = np.arange(len(locations))  # label locations
width = 0.2  # width of the bars

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
ax.axhline(
    y=10, color="red", linestyle="--", linewidth=1.5, label="10% threshold", alpha=0.4
)
ax.axhline(y=-10, color="red", linestyle="--", linewidth=1.5, alpha=0.4)
ax.axhline(y=0, color="k", linewidth=1.5, alpha=1)

# Plot each metric as a bar set
for i, (metric, label) in enumerate(metrics.items()):
    values = [results[loc][metric] for loc in locations]
    # ax.bar(x + i * width - width, values, width, label=metric, color=colors[i])
    offset = i * width - width  # adjust bar position
    ax.bar(x + offset, values, width, label=label, color=colors[i])

    if metric == "Net_metric" or metric == "Net_absolute_metric_2":
        # Draw invisible bars
        # bars = ax.bar(
        #     x + offset,
        #     values,
        #     width,
        #     label=metric,
        #     color="none",
        #     edgecolor="black",
        #     linestyle="dotted",
        # )

        # Draw top lines for absolute values
        for xi, val in zip(x + offset, values):
            ax.hlines(
                abs(val),
                xi - width / 2,
                xi + width / 2,
                color=colors[i],
                linewidth=2,
                linestyles="dotted",
            )
    # else:
    # Regular bars

# Labeling
ax.set_xlabel("Location")
ax.set_ylabel("Error Percent")
ax.set_title("Error Percent by Location and Metric")
ax.set_xticks(x)
ax.set_xticklabels(locations)
ax.legend()
print(min([(results[loc][metric]) for loc in locations for metric in metrics.keys()]))
min_y = (
    int(min([results[loc][metric] for loc in locations for metric in metrics.keys()]))
    - 1
)
max_y = (
    int(max([results[loc][metric] for loc in locations for metric in metrics.keys()]))
    + 1
)
min_y = min(min_y, -10)
max_y = max(max_y, 10)
ax.set_yticks(np.arange(min_y - 1, max_y + 1, 1))

ax.grid(True, axis="y", linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("analysis/figures/error_percent_by_location.png", dpi=300)
plt.show()

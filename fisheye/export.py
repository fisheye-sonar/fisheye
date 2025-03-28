import os
from datetime import datetime

import pandas as pd


def to_csv(data, out_dir):
    """Export inference results to CSV file."""
    out_file = os.path.join(
        out_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    )

    df = pd.json_normalize(data, "tracks", ["file", "counts"])
    df.to_csv(out_file, index=False)

    print(f"Exported results to {out_file}")


def to_text(data, out_dir):
    """Export inference results to text file."""

    out_file = os.path.join(
        out_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    )

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

    default_values = {
        "File": 1,
        "Total": 1,
        "Frame#": 0,
        "Dir": "Up",
        "R (m)": 0.0,
        "Theta": 0.0,
        "L(cm)": 0.0,
        "dR(cm)": 0.0,
        "L/dR": 0.0,
        "Aspect": 0.0,
        "Time": "00:00:00",
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Latitude": "N 00 d  0.00000 m",
        "Longitude": "E 000 d  0.00000 m",
        "Pan": 0.0,
        "Tilt": 0.0,
        "Roll": 0.0,
        "Species": "Unknown",
        "Motion": "Running <->",
        "Q": 5,
        "N": 1,
        "Comment": "",
    }

    header_line = "  ".join(f"{h:<10}" for h in headers)
    separator_line = "-" * len(header_line)

    with open(out_file, "w") as f:
        f.write(header_line + "\n")
        f.write(separator_line + "\n")

        for row in data:
            # Merge row with defaults to ensure all keys are present
            row_data = {**default_values, **row}
            row_line = "  ".join(f"{str(row_data[h]):<10}" for h in headers)
            f.write(row_line + "\n")

    print(f"Exported results to {out_file}")

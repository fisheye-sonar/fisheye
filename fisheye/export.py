import os
from datetime import datetime
from typing import Dict, Callable, Any

import pandas as pd

from fisheye.enums import ExportType


def to_csv(data, out_dir):
    """Export inference results to CSV file.

    Two CSVs are generated:
    1. A detailed CSV with all track counts.
    2. A summary CSV with net counts per ARIS/DDF file.
    """
    out_file = os.path.join(out_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    flattened_data = [item for sublist in data for item in sublist]
    df = pd.DataFrame(flattened_data)
    # Save off all track counts to CSV
    df.to_csv(out_file + ".csv", index=False)

    summary_df = df.groupby(["file_name", "direction"]).size().unstack(fill_value=0)
    summary_df = summary_df.rename(
        columns={"left": "abs_left", "right": "abs_right"}
    ).reset_index()
    summary_df["net_count"] = abs(summary_df["abs_left"] - summary_df["abs_right"])

    # Save off the absolute left counts, absolute right counts, absolute net counts for each ARIS/DDF file to CSV
    summary_df.to_csv(out_file + "_summary.csv", index=False)

    print(f"Exported results to {out_dir}")


# Add any new export functions here
EXPORT_FUNCTIONS: Dict[ExportType, Callable[[Any, str], None]] = {
    ExportType.CSV: to_csv,
}


def get_exporter(export_type: ExportType | str) -> Callable[[Any, str], None]:
    """Retrieve the appropriate export function."""

    if isinstance(export_type, str):
        export_type = ExportType(export_type)

    return EXPORT_FUNCTIONS.get(export_type)

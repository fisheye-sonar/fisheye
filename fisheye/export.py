import os
from datetime import datetime
from typing import Dict, Callable, Any

import pandas as pd

from fisheye.enums import ExportType


def to_csv(data, out_dir):
    """Export inference results to CSV file."""
    out_file = os.path.join(
        out_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    )

    df = pd.json_normalize(data, "tracks", ["file", "counts"])
    df.to_csv(out_file, index=False)

    print(f"Exported results to {out_file}")


# Add any new export functions here
EXPORT_FUNCTIONS: Dict[ExportType, Callable[[Any, str], None]] = {
    ExportType.CSV: to_csv,
}


def get_exporter(export_type: ExportType | str) -> Callable[[Any, str], None]:
    """Retrieve the appropriate export function."""

    if isinstance(export_type, str):
        export_type = ExportType(export_type)

    return EXPORT_FUNCTIONS.get(export_type)

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

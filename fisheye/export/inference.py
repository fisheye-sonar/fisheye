import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Callable, Any, Union, List

import pandas as pd
import structlog

from fisheye.configs.datasets import ARISMetadata
from fisheye.enums import ExportType
from fisheye.utils import get_unwarped_distance_and_theta

logger = structlog.get_logger()


def to_detailed_csv(data, out_dir, job_id: str = None, **kwargs):
    """Export inference results to CSV file.

    Users can configure the distance offset via kwargs:
        distance_offset: float, default 0.0
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    job_suffix = f"_{job_id}" if job_id else ""
    out_file = os.path.join(out_dir, f"{timestamp}{job_suffix}")

    flattened_data = [item for sublist in data if sublist for item in sublist]
    if not flattened_data:
        logger.warning("No counts were found in the provided data. Nothing to export.")
        return

    out_file = out_file + "_" + Path(flattened_data[0].get("Source.Name")).stem + ".csv"
    distance_offset = kwargs.get("distance_offset", 0.0)

    # Flatten metadata into each row
    expanded_data = []
    for item in flattened_data:
        new_item = item.copy()
        meta = new_item["metadata"]
        if isinstance(meta, ARISMetadata):
            new_item.update(meta.__dict__)
        expanded_data.append(new_item)

    df = pd.DataFrame(expanded_data)

    # Calculate unwarped distance if bbox exists
    if "bbox" in df.columns and not df["bbox"].isna().all():
        df[["R (m)", "Theta"]] = df.apply(
            get_unwarped_distance_and_theta, axis=1, result_type="expand"
        )

    df["R (m)"] += distance_offset

    # Extract date from first source name (source names are all the same)
    source_name = df.loc[0, "Source.Name"]
    m = re.search(r"\d{4}-\d{2}-\d{2}", source_name)
    formatted_date = (
        pd.to_datetime(m.group(0), format="%Y-%m-%d").strftime("%m-%d-%Y")
        if m
        else None
    )
    df["Date"] = formatted_date

    # Sort by frame index in ascending order
    df = df.sort_values(by="Frame#")

    # Column ordering
    base_cols = ["Source.Name", "Frame#", "Dir", "R (m)", "Theta", "Date", "ID"]

    # Columns from ARISMetadata
    meta_cols = [c for c in df.columns if c not in base_cols and c not in ["bbox"]]

    # Remaining columns (catch-all for any other unexpected fields)
    remaining_cols = [c for c in df.columns if c not in base_cols + meta_cols]

    final_cols = base_cols + meta_cols + remaining_cols
    df = df[final_cols]
    df.drop(columns=["metadata"], inplace=True)

    with open(out_file, "w") as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())

    logger.info("exported_detailed_csv", output_dir=out_file)


def to_summary_csv(data, out_dir, job_id: str = None, **kwargs):
    """Export inference results to CSV file, ensuring all files are represented.

    Args:
        data (dict): Dictionary of inference results.
        out_dir (str): Output directory for CSV files.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    job_suffix = f"_{job_id}" if job_id else ""
    out_file = os.path.join(out_dir, f"{timestamp}{job_suffix}_summary.csv")

    flattened_data = [item for sublist in data if sublist for item in sublist]

    if not flattened_data:
        logger.warning("No counts were found in the provided data. Nothing to export.")
        return

    df = pd.DataFrame(flattened_data)
    df["ID"] = df.get("ID", pd.NA)
    df["Dir"] = df.get("Dir", pd.NA)

    # Get all unique file names upfront
    all_files = df["Source.Name"].unique()

    # Only valid rows for counting
    valid_rows = df.dropna(subset=["ID", "Dir"])

    if not valid_rows.empty:
        direction_counts = (
            valid_rows.groupby(["Source.Name", "ID", "Dir"])
            .size()
            .unstack(fill_value=0)
        )

        direction_counts = direction_counts.reindex(
            columns=["Up", "Down"], fill_value=0
        )

        direction_counts["absolute_up"] = (
            direction_counts["Up"] > direction_counts["Down"]
        ).astype(int)
        direction_counts["absolute_down"] = (
            direction_counts["Down"] > direction_counts["Up"]
        ).astype(int)

        file_counts = direction_counts.groupby("Source.Name")[
            ["absolute_up", "absolute_down"]
        ].sum()

    else:
        file_counts = pd.DataFrame(
            columns=["absolute_up", "absolute_down"], index=all_files
        ).fillna(0)

    # Ensure all files are represented
    for file in all_files:
        if file not in file_counts.index:
            file_counts.loc[file] = {"absolute_up": 0, "absolute_down": 0}

    file_counts["net_count"] = file_counts["absolute_up"] - file_counts["absolute_down"]
    final_result = file_counts.reset_index().rename(columns={"index": "Source.Name"})

    with open(out_file, "w") as f:
        final_result.to_csv(out_file, index=False)
        f.flush()
        os.fsync(f.fileno())

    logger.info(f"exported_summary_csv", output_dir=out_file)


def to_fc_txt(data, out_dir, **kwargs):
    """Export inference results to FC TXT file.

    This exporter replicates the format of the FC TXT output generated by the ARISFish software from Sound Metrics.
    Args:
        data (dict): Dictionary of inference results.
        out_dir (str): Output directory for TXT file(s).
    """
    flattened_data = [item for sublist in data if sublist for item in sublist]
    if not flattened_data:
        logger.warning(f"No counts were found in the provided data. Nothing to export.")
        return

    distance_offset = kwargs.get("distance_offset", 0.0)
    df = pd.DataFrame(flattened_data)

    if not df["bbox"].isna().all():
        # Calculate the distance from the sonar camera to the fish in an unwarped frame
        df[["R (m)", "Theta"]] = df.apply(
            get_unwarped_distance_and_theta, axis=1, result_type="expand"
        )

    title = "*** Manual Marking (Manual Sizing: Q = Quality, N = Repeat Count) ***"

    # Add any new header fields to the end of the list (must maintain current order)
    # Also make sure to add any new header fields to `row_data`
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

    col_width = 2
    header_line = "  ".join(f"{h:<{col_width}}" for h in headers)
    separator_line = "-" * len(header_line)

    for file_name, group_df in df.groupby("Source.Name"):
        match = re.search(r"(\d{4}-\d{2}-\d{2})", file_name)
        # Attempt to get date from filename
        date = match.group(1) if match else datetime.now().strftime("%Y-%m-%d")
        group_df_sorted = group_df.sort_values(by="Frame#")

        row_data = {
            "File": 1,
            "Total": 0,
            "Frame#": 0,
            "Dir": "",
            "R (m)": 0.0,
            "Theta": 0.0,
            "L(cm)": 0.0,  # TODO (MVH) - update to use our length estimations
            "dR(cm)": 0.0,
            "L/dR": 0.0,
            "Aspect": 0.0,
            "Time": "00:00:00",
            "Date": date,
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

        file_stem = Path(file_name).stem
        out_file = os.path.join(out_dir, f"FCe_{file_stem}_ID_.txt")

        lines = [title + "\n\n", header_line + "\n", separator_line + "\n"]
        for _, row in group_df_sorted.iterrows():
            bbox = row.get("bbox")
            has_data = (
                bbox is not None
                and len(bbox) > 0
                and "R (m)" in row
                and pd.notna(row["R (m)"])
            )

            if has_data:
                row_data = {
                    "File": 1,
                    "Total": row_data.get("Total", 0) + 1,
                    "Frame#": row.get("Frame#", 0),
                    "Dir": row.get("Dir"),
                    "R (m)": row.get("R (m)", 0)
                    + distance_offset,  # MVH: Apply offset to marker placement in ARISFish to avoid covering the fish.
                    "Theta": row.get("Theta", 0),
                    "L(cm)": 0.0,
                    "dR(cm)": 0.0,
                    "L/dR": 0.0,
                    "Aspect": 0.0,
                    "Time": "00:00:00",
                    "Date": date,
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

                row_line = "  ".join(f"{str(row_data[h]):<10}" for h in headers)
            else:
                row_line = "  ".join(["" for _ in headers])

            lines.append(row_line + "\n")

        with open(out_file, "w") as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())

        logger.info(f"exported_fc_txt", output_dir=out_file)


def to_mot_txt(data, output_dir, filename, **kwargs):
    """Export inference results to MOT file(s). Expects data to be in MOT output already."""
    out_path = os.path.join(output_dir, filename + ".txt")

    mot_lines = []
    for row in data:
        # Convert frame number to int (if needed) and format row to MOT string
        if row:
            mot_line = "{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{}".format(
                row.get("frame"),
                row.get("id"),
                row.get("bb_left"),
                row.get("bb_top"),
                row.get("bb_width"),
                row.get("bb_height"),
                row.get("conf"),
                row.get("x"),
                row.get("y"),
                row.get("z"),
            )
            mot_lines.append(mot_line + "\n")
        else:
            mot_lines.append("\n")

    with open(out_path, "w") as f:
        f.writelines(mot_lines)
        f.flush()
        os.fsync(f.fileno())

    logger.info(f"exported_mot_txt", output_dir=out_path)


# Add any new export functions here
EXPORT_FUNCTIONS: Dict[ExportType, Callable[[Any, str], None]] = {
    ExportType.DETAILED_CSV: to_detailed_csv,
    ExportType.SUMMARY_CSV: to_summary_csv,
    ExportType.FC: to_fc_txt,
    ExportType.MOT: to_mot_txt,
}


def get_exporter(export_type: Union[ExportType, str]) -> Callable[[Any, str], None]:
    """Retrieve the appropriate export function."""

    if isinstance(export_type, str):
        export_type = ExportType(export_type)

    return EXPORT_FUNCTIONS.get(export_type)


def save_to_disk(
    results,
    output_dir,
    export_types: Union[List[ExportType], ExportType],
    job_id: str,
    distance_offset: Union[int, float],
) -> None:
    """Save results to disk."""
    if not results or all(len(sublist) == 0 for sublist in results):
        logger.warning(f"No counts were found in the provided data. Nothing to export.")
        return

    if not isinstance(export_types, list):
        export_types = [export_types]

    for export_option in export_types:
        exporter = get_exporter(export_option)

        if export_option in [ExportType.DETAILED_CSV, ExportType.SUMMARY_CSV]:
            exporter(
                results, output_dir, job_id=job_id, distance_offset=distance_offset
            )

        else:
            exporter(results, output_dir, distance_offset=distance_offset)


def parse_export_options(options: List[str]) -> List[ExportType]:
    export_types = []
    for option in options:
        try:
            export_types.append(ExportType[option.strip().upper()])
        except KeyError as e:
            raise ValueError(f"Invalid export type: {e.args[0]}")

    return export_types

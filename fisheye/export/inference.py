import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, List, Optional

import pandas as pd
import structlog

from fisheye.enums import ExportType
from fisheye.export.constants import (
    FC_DEFAULT_LENGTH_CM,
    FC_DEFAULT_DR_CM,
    FC_DEFAULT_L_OVER_R,
    FC_DEFAULT_ASPECT,
    FC_DEFAULT_TIME,
    FC_DEFAULT_LATITUDE,
    FC_DEFAULT_LONGITUDE,
    FC_DEFAULT_PAN,
    FC_DEFAULT_TILT,
    FC_DEFAULT_ROLL,
    FC_DEFAULT_SPECIES,
    FC_DEFAULT_MOTION,
    FC_DEFAULT_QUALITY,
    FC_DEFAULT_REPEAT_COUNT,
    FC_DEFAULT_COMMENT,
)
from fisheye.utils import get_unwarped_distance_and_theta

logger = structlog.get_logger()


class BaseInferenceExporter(ABC):
    """Abstract base class for export strategies."""

    def __init__(
        self,
        output_dir: str,
        job_id: Optional[str] = None,
        distance_offset: float = 0.0,
    ):
        self.output_dir = output_dir
        self.job_id = job_id
        self.distance_offset = distance_offset
        self.timestamp = datetime.now().strftime("%Y-%m-%d")
        self.job_suffix = f"_{job_id}" if job_id else ""

    def _prepare_dataframe(self, data: List[List[Dict]]) -> Optional[pd.DataFrame]:
        """Flattens data and prepares DataFrame with common fields."""
        flattened_data = [item for sublist in data if sublist for item in sublist]
        if not flattened_data:
            return None

        # Optimize flattening: Convert metadata objects to dicts upfront
        processed_data = []
        for item in flattened_data:
            # Shallow copy to avoid modifying original
            new_item = item.copy()
            meta = new_item.get("metadata")
            if meta:
                if hasattr(meta, "__dict__"):
                    new_item.update(meta.__dict__)
                elif isinstance(meta, dict):
                    new_item.update(meta)
            processed_data.append(new_item)

        df = pd.DataFrame(processed_data)

        # Calculate the distance from the sonar camera to the fish in an unwarped frame
        if "bbox" in df.columns and not df["bbox"].isna().all():
            df[["R (m)", "Theta"]] = df.apply(
                get_unwarped_distance_and_theta, axis=1, result_type="expand"
            )

        # Ensure columns exist
        if "R (m)" not in df.columns:
            df["R (m)"] = 0.0
        if "Theta" not in df.columns:
            df["Theta"] = 0.0

        # Apply offset
        df["R (m)"] += self.distance_offset
        df["R (m)"] = df["R (m)"].round(2)

        return df

    @abstractmethod
    def export(self, data: List[List[Dict]]) -> None:
        """Execute the export process."""
        pass


class DetailedCSVExporter(BaseInferenceExporter):
    def export(self, data: List[List[Dict]]) -> None:
        df = self._prepare_dataframe(data)
        if df is None or df.empty:
            logger.warning(
                "No counts were found in the provided data. Nothing to export."
            )
            return

        stem = Path(str(df.iloc[0]["Source.Name"])).stem
        out_file = os.path.join(
            self.output_dir, f"{self.timestamp}{self.job_suffix}_{stem}.csv"
        )

        source_name = str(df.iloc[0]["Source.Name"])
        m = re.search(r"\d{4}-\d{2}-\d{2}", source_name)
        formatted_date = (
            pd.to_datetime(m.group(0), format="%Y-%m-%d").strftime("%m-%d-%Y")
            if m
            else None
        )
        df["Date"] = formatted_date

        df = df.sort_values(by="Frame#")

        # Column ordering
        base_cols = ["Source.Name", "Frame#", "Dir", "R (m)", "Theta", "Date", "ID"]
        meta_cols = [
            c
            for c in df.columns
            if c not in base_cols and c not in ["bbox", "metadata"]
        ]
        remaining_cols = [
            c
            for c in df.columns
            if c not in base_cols + meta_cols + ["bbox", "metadata"]
        ]

        final_cols = base_cols + meta_cols + remaining_cols
        df = df[final_cols]

        with open(out_file, "w") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())

        logger.info("exported_detailed_csv", output_dir=out_file)


class SummaryCSVExporter(BaseInferenceExporter):
    def export(self, data: List[List[Dict]]) -> None:
        flattened_data = [item for sublist in data if sublist for item in sublist]
        if not flattened_data:
            logger.warning(
                "No counts were found in the provided data. Nothing to export."
            )
            return

        out_file = os.path.join(
            self.output_dir, f"{self.timestamp}{self.job_suffix}_summary.csv"
        )

        df = pd.DataFrame(flattened_data)
        df["ID"] = df.get("ID", pd.NA)
        df["Dir"] = df.get("Dir", pd.NA)

        all_files = df["Source.Name"].unique()
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

        # Ensure all files represented
        for file in all_files:
            if file not in file_counts.index:
                file_counts.loc[file] = {"absolute_up": 0, "absolute_down": 0}

        file_counts["net_count"] = (
            file_counts["absolute_up"] - file_counts["absolute_down"]
        )
        final_result = file_counts.reset_index().rename(
            columns={"index": "Source.Name"}
        )

        with open(out_file, "w") as f:
            final_result.to_csv(out_file, index=False)
            f.flush()
            os.fsync(f.fileno())

        logger.info("exported_summary_csv", output_dir=out_file)


class FCExporter(BaseInferenceExporter):
    def export(self, data: List[List[Dict]]) -> None:
        df = self._prepare_dataframe(data)
        if df is None or df.empty:
            logger.warning(
                "No counts were found in the provided data. Nothing to export."
            )
            return

        title = "*** Manual Marking (Manual Sizing: Q = Quality, N = Repeat Count) ***"
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

        # Set default values for missing columns
        defaults = {
            "File": 1,
            "Total": 0,  # Calculated per group
            "Frame#": 0,
            "Dir": "",
            "R (m)": 0.0,
            "Theta": 0.0,
            "L(cm)": FC_DEFAULT_LENGTH_CM,
            "dR(cm)": FC_DEFAULT_DR_CM,
            "L/dR": FC_DEFAULT_L_OVER_R,
            "Aspect": FC_DEFAULT_ASPECT,
            "Time": FC_DEFAULT_TIME,
            "Latitude": FC_DEFAULT_LATITUDE,
            "Longitude": FC_DEFAULT_LONGITUDE,
            "Pan": FC_DEFAULT_PAN,
            "Tilt": FC_DEFAULT_TILT,
            "Roll": FC_DEFAULT_ROLL,
            "Species": FC_DEFAULT_SPECIES,
            "Motion": FC_DEFAULT_MOTION,
            "Q": FC_DEFAULT_QUALITY,
            "N": FC_DEFAULT_REPEAT_COUNT,
            "Comment": FC_DEFAULT_COMMENT,
        }

        for file_name, group_df in df.groupby("Source.Name"):
            match = re.search(r"(\d{4}-\d{2}-\d{2})", file_name)
            date = match.group(1) if match else datetime.now().strftime("%Y-%m-%d")

            group_df = group_df.sort_values(by="Frame#").copy()

            # Filter valid rows
            valid_mask = group_df["R (m)"].notna()
            group_df = group_df[valid_mask]

            if group_df.empty:
                lines = [title + "\n\n", header_line + "\n", separator_line + "\n"]
            else:
                # Populate defaults
                for col, val in defaults.items():
                    if col not in group_df.columns:
                        group_df[col] = val

                group_df["Date"] = date
                group_df["Total"] = range(1, len(group_df) + 1)

                lines = [title + "\n\n", header_line + "\n", separator_line + "\n"]

                # Ensure all header columns exist
                for h in headers:
                    if h not in group_df.columns:
                        group_df[h] = ""

                # Select only header columns in order
                export_df = group_df[headers]

                # Convert to string and pad
                rows = [
                    "  ".join(f"{str(val):<10}" for val in row)
                    for row in export_df.values
                ]
                lines.extend(r + "\n" for r in rows)

            file_stem = Path(str(file_name)).stem
            out_file = os.path.join(self.output_dir, f"FCe_{file_stem}_ID_.txt")

            with open(out_file, "w") as f:
                f.writelines(lines)
                f.flush()
                os.fsync(f.fileno())

            logger.info("exported_fc_txt", output_dir=out_file)


class MOTExporter(BaseInferenceExporter):
    def __init__(
        self,
        output_dir: str,
        job_id: Optional[str] = None,
        distance_offset: float = 0.0,
        filename: Optional[str] = None,
    ):
        super().__init__(output_dir, job_id, distance_offset)
        self.filename = filename

    def export(self, data: List[Dict]) -> None:
        # Use filename if available, else construct one
        fname = (
            self.filename if self.filename else f"{self.timestamp}{self.job_suffix}_mot"
        )
        out_path = os.path.join(self.output_dir, fname + ".txt")

        mot_lines = []
        for row in data:
            if row:
                # MOT format: frame, id, left, top, width, height, conf, x, y, z
                mot_line = "{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{}".format(
                    row.get("frame", -1),
                    row.get("id", -1),
                    row.get("bb_left", 0),
                    row.get("bb_top", 0),
                    row.get("bb_width", 0),
                    row.get("bb_height", 0),
                    row.get("conf", 0),
                    row.get("x", -1),
                    row.get("y", -1),
                    row.get("z", -1),
                )
                mot_lines.append(mot_line + "\n")
            else:
                mot_lines.append("\n")

        with open(out_path, "w") as f:
            f.writelines(mot_lines)
            f.flush()
            os.fsync(f.fileno())

        logger.info("exported_mot_txt", output_dir=out_path)


def get_exporter(
    export_type: Union[ExportType, str],
    output_dir: str,
    job_id: str,
    distance_offset: float = 0.0,
    **kwargs,
):
    """Factory to create exporters."""
    if isinstance(export_type, str):
        try:
            export_type = ExportType(export_type)
        except ValueError:
            export_type = ExportType[export_type.upper()]

    if export_type == ExportType.DETAILED_CSV:
        return DetailedCSVExporter(output_dir, job_id, distance_offset)

    elif export_type == ExportType.SUMMARY_CSV:
        return SummaryCSVExporter(output_dir, job_id, distance_offset)

    elif export_type == ExportType.FC:
        return FCExporter(output_dir, job_id, distance_offset)

    elif export_type == ExportType.MOT:
        return MOTExporter(
            output_dir, job_id, distance_offset, filename=kwargs.get("filename")
        )

    else:
        raise ValueError(f"Unsupported export type: {export_type}")


def save_to_disk(
    results: List[List[Dict]],
    output_dir: str,
    export_types: Union[List[ExportType], ExportType],
    job_id: str,
    distance_offset: Union[int, float],
) -> None:
    """Save results to disk using configured exporters."""
    if not results or all(len(sublist) == 0 for sublist in results):
        logger.warning("No counts were found in the provided data. Nothing to export.")
        return

    if not isinstance(export_types, list):
        export_types = [export_types]

    for export_option in export_types:
        try:
            exporter = get_exporter(
                export_option, output_dir, job_id, float(distance_offset)
            )
            exporter.export(results)
        except Exception as e:
            logger.error(
                "export_failed", export_type=export_option, error=str(e), exc_info=True
            )


def parse_export_options(options: List[str]) -> List[ExportType]:
    export_types = []
    for option in options:
        try:
            export_types.append(ExportType[option.strip().upper()])
        except KeyError as e:
            raise ValueError(f"Invalid export type: {e.args[0]}")

    return export_types

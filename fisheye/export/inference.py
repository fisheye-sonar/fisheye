import os
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, List, Optional

import numpy as np
import pandas as pd
import structlog

from fisheye.enums import ExportType
from fisheye.export.schema import FC_SCHEMA, get_zero_and_length_points
from fisheye.utils import (
    get_unwarped_distance_and_theta,
    convert_pixels_to_coords_meters,
)

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

    @staticmethod
    def _flatten_list(data: List[List[Dict]]) -> List:
        """Flattens lists to process easier."""
        flattened_data = [item for sublist in data if sublist for item in sublist]
        if not flattened_data:
            return []

        return flattened_data

    @staticmethod
    def _has_data_to_export(data: Union[List, pd.DataFrame]) -> bool:
        """Return True if data contains exportable entries."""
        is_empty = (
            data is None
            or (isinstance(data, pd.DataFrame) and data.empty)
            or (isinstance(data, list) and len(data) == 0)
        )

        if is_empty:
            logger.warning(
                "no_data_to_export",
                message="No counts were found in the provided data. Nothing to export.",
            )
            return False

        return True

    def _prepare_dataframe(self, data: List[List[Dict]]) -> Optional[pd.DataFrame]:
        """Flattens data and prepares DataFrame with common fields."""
        flattened_data = self._flatten_list(data)

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

        if "metadata" in df.columns:
            # Drop metadata column since we expanded it into individual columns
            df = df.drop(columns=["metadata"])

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
    """Export counts to detailed CSV format."""

    def export(self, data: List[List[Dict]]) -> None:
        df = self._prepare_dataframe(data)
        if not self._has_data_to_export(df):
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
        meta_cols = [c for c in df.columns if c not in base_cols]

        final_cols = base_cols + meta_cols
        df = df[final_cols]

        with open(out_file, "w") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())

        logger.info("exported_detailed_csv", output_dir=out_file)


class SummaryCSVExporter(BaseInferenceExporter):
    """Export counts to summary CSV format."""

    def export(self, data: List[List[Dict]]) -> None:
        flattened_data = self._flatten_list(data)
        if not self._has_data_to_export(flattened_data):
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
    """Export counts to FC format."""

    def export(self, data: List[List[Dict]]) -> None:
        df = self._prepare_dataframe(data)
        if not self._has_data_to_export(df):
            return

        title = "*** Manual Marking (Manual Sizing: Q = Quality, N = Repeat Count) ***"

        headers = list(FC_SCHEMA)
        column_widths = {k: v.width for k, v in FC_SCHEMA.items()}
        defaults = {k: v.default for k, v in FC_SCHEMA.items()}

        header_line = "".join(f"{h:>{column_widths[h]}}" for h in headers)
        separator_line = "-" * len(header_line)

        for file_name, group_df in df.groupby("Source.Name"):
            match = re.search(r"(\d{4}-\d{2}-\d{2})", file_name)
            date = match.group(1) if match else datetime.now().strftime("%Y-%m-%d")

            group_df = group_df.sort_values(by="Frame#").copy()

            # Filter valid rows
            valid_mask = group_df["R (m)"].notna()
            group_df = group_df[valid_mask]

            if group_df.empty or group_df is None or group_df["Frame#"].isna().all():
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

                for row in export_df.itertuples(index=False):
                    lines.append(
                        "".join(
                            f"{str(val):>{column_widths[headers[i]]}}"
                            for i, val in enumerate(row)
                        )
                        + "\n"
                    )

            file_stem = Path(str(file_name)).stem
            out_file = os.path.join(self.output_dir, f"FCe_{file_stem}_ID_.txt")

            with open(out_file, "w") as f:
                f.writelines(lines)
                f.flush()
                os.fsync(f.fileno())

            logger.info("exported_fc_txt", output_dir=out_file)


class MOTExporter(BaseInferenceExporter):
    """Export counts to MOT format."""

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


class XMLExporter(BaseInferenceExporter):
    """Export head/tail measurements to XML format."""

    def __init__(
        self,
        output_dir: str,
        job_id: Optional[str] = None,
        distance_offset: float = 0.0,
        upstream_direction: Optional[str] = None,
    ):
        super().__init__(output_dir, job_id, distance_offset)
        self.upstream_direction = upstream_direction

    def export(self, data: List[List[Dict]]) -> None:
        flattened_data = self._flatten_list(data)

        root = ET.Element("MarkedFishMeasurements")

        for d in flattened_data:
            source_name = Path(d.get("Source.Name")).stem
            output_path = os.path.join(self.output_dir, f"FCe_{source_name}_ID_.xml")

            marked = ET.SubElement(
                root,
                "MarkedFishMeasurement",
                {
                    "FishID": str(d.get("ID", 0)),
                    "FrameIndex": str(d.get("Frame#", 0)),
                },
            )

            global_coords_px = d.get("global_coords_px", [])
            if not global_coords_px:
                continue

            world_points = convert_pixels_to_coords_meters(
                np.array(global_coords_px), d["metadata"]
            )
            left_point, right_point = world_points[:2]

            # Figure out point order so first point is always length=0 (head)
            zero_point, length_point = get_zero_and_length_points(
                self.upstream_direction, d["Dir"], left_point, right_point
            )

            ET.SubElement(
                marked,
                "FishMeasureNode",
                {
                    "WorldPointX": f"{zero_point[0]}",
                    "WorldPointY": f"{zero_point[1]}",
                    "Length": "0",
                },
            )

            ET.SubElement(
                marked,
                "FishMeasureNode",
                {
                    "WorldPointX": f"{length_point[0]}",
                    "WorldPointY": f"{length_point[1]}",
                    "Length": f'{d["L(cm)"]}',
                },
            )

        tree = ET.ElementTree(root)

        output_path = Path(output_path)

        tree.write(
            output_path,
            encoding="utf-8",
            xml_declaration=True,
        )

        logger.info("exported_xml", output_dir=output_path)


def get_exporter(
    export_type: Union[ExportType, str],
    output_dir: str,
    job_id: str,
    distance_offset: float = 0.0,
    upstream_direction: str = "left",
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

    elif export_type == ExportType.XML:
        return XMLExporter(output_dir, job_id, distance_offset, upstream_direction)

    else:
        raise ValueError(f"Unsupported export type: {export_type}")


def save_to_disk(
    results: List[List[Dict]],
    output_dir: str,
    export_types: Union[List[ExportType], ExportType],
    job_id: str,
    distance_offset: Union[int, float],
    upstream_direction: str,
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
                export_option,
                output_dir,
                job_id,
                float(distance_offset),
                upstream_direction,
            )
            exporter.export(results)
        except Exception as e:
            logger.error(
                "export_failed", export_type=export_option, error=str(e), exc_info=True
            )


def parse_export_options(options: List[str]) -> List[ExportType]:
    """Parse export options from CLI."""
    export_types = []
    for option in options:
        try:
            export_types.append(ExportType[option.strip().upper()])
        except KeyError as e:
            raise ValueError(f"Invalid export type: {e.args[0]}")

    return export_types

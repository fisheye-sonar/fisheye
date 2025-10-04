from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader
from tqdm import tqdm

from parse_csv import parse_csv_file
from parse_fc import parse_fc_file
from col import col


@dataclass(frozen=True)
class Config:
    fc_dir: Path
    csv_dir: Path
    aris_dir: Optional[Path]
    report_figures_dir: Path
    report_fp: Path
    fc_label: str = "Corrected"
    csv_label: str = "Predicted"
    near_bank_cutoff_distance: float = 16.0
    generate_coverage: bool = True
    beam_width_dir: Optional[Path] = None


@dataclass
class DataIndex:
    fc_files: list[str]
    csv_files: list[Optional[str]]
    aris_with_csv: list[Optional[str]]
    aris_with_fc: list[Optional[str]]
    clip_datetimes: list[datetime]
    all_aris: list[str]
    all_have_aris: bool


@dataclass
class ClipComparison:
    file_completely_correct: bool
    exact_matches: int
    in_fc_upstream: int
    in_fc_downstream: int
    in_csv_upstream: int
    in_csv_downstream: int
    missing_in_fc: int
    missing_in_csv: int
    missing_in_fc_perc: float
    missing_in_csv_perc: float
    missing_in_fc_upstream: int
    missing_in_fc_downstream: int
    missing_in_csv_upstream: int
    missing_in_csv_downstream: int


@dataclass
class CombinedData:
    comparisons: list[ClipComparison]
    fc_records: list[dict]
    csv_records: list[dict]
    datetimes: list[datetime]
    missing_csv_count: int


@dataclass
class CoverageResult:
    coverage: Optional[float]
    all_aris_intervals: list[tuple[datetime, datetime]]
    analysis_intervals: list[Optional[tuple[datetime, datetime]]]


@dataclass
class AggregateStats:
    csv_count: int
    fc_count: int
    total_counts_csv: int
    total_counts_fc: int
    total_counts_error: int
    total_counts_error_pct: float
    near_csv_count: int
    near_fc_count: int
    far_csv_count: int
    far_fc_count: int
    precision: float
    recall: float
    total_crossings_csv_upstream: int
    total_crossings_fc_upstream: int
    total_crossings_csv_downstream: int
    total_crossings_fc_downstream: int
    error_upstream: int
    error_downstream: int
    number_of_correct_matches: int
    missing_in_fc_total: int
    missing_in_csv_total: int
    clipwise_net_fc: list[int]
    clipwise_net_csv: list[int]
    min_counts: int
    max_counts: int
    min_distance: float
    max_distance: float
    valid_csvs: int
    csvs_missing: int
    valid_fcs: int
    fcs_missing: int
    valid_ariss: int
    ariss_missing: int


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Compare FCe files against model CSV outputs and generate a PDF summary.",
    )
    parser.add_argument(
        "--fc-dir",
        type=Path,
        required=True,
        help="Directory containing corrected FCe files.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Directory containing model CSV files.",
    )
    parser.add_argument(
        "--aris-dir",
        type=Path,
        default=None,
        help="Directory containing ARIS files (optional).",
    )
    parser.add_argument(
        "--report-figures-dir",
        type=Path,
        required=True,
        help="Directory to write generated plot images.",
    )
    parser.add_argument(
        "--report-fp",
        type=Path,
        required=True,
        help="Path to the output summary PDF file.",
    )
    parser.add_argument(
        "--beam-width-dir",
        type=Path,
        default=None,
        help="Directory containing beam width metadata.",
    )
    parser.add_argument(
        "--fc-label", default="Corrected", help="Label for the corrected FC dataset."
    )
    parser.add_argument(
        "--csv-label", default="Predicted", help="Label for the model CSV dataset."
    )
    parser.add_argument(
        "--near-bank-cutoff-distance",
        type=float,
        default=16.0,
        help="Distance threshold (meters) for near-bank aggregation.",
    )
    parser.add_argument(
        "--generate-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable ARIS coverage calculations.",
    )

    args = parser.parse_args(argv)

    return Config(
        fc_dir=args.fc_dir,
        csv_dir=args.csv_dir,
        aris_dir=args.aris_dir,
        report_figures_dir=args.report_figures_dir,
        report_fp=args.report_fp,
        fc_label=args.fc_label,
        csv_label=args.csv_label,
        near_bank_cutoff_distance=args.near_bank_cutoff_distance,
        generate_coverage=args.generate_coverage,
        beam_width_dir=args.beam_width_dir,
    )


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def parse_datetime_from_tokens(tokens: Sequence[str]) -> datetime:
    date = tokens[-2]
    time = tokens[-1]
    return datetime.strptime(f"{date} {time}", "%Y-%m-%d %H%M%S")


def discover_data(config: Config) -> DataIndex:
    if not config.fc_dir.exists():
        raise FileNotFoundError(f"FC directory not found: {config.fc_dir}")
    if not config.csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {config.csv_dir}")

    fc_files = sorted(
        fn.name
        for fn in config.fc_dir.iterdir()
        if fn.is_file() and fn.name.startswith("FCe_") and fn.suffix == ".txt"
    )

    csv_files = [None] * len(fc_files)
    aris_with_csv = [None] * len(fc_files)
    aris_with_fc = [None] * len(fc_files)

    all_csvs = sorted(
        fn.name
        for fn in config.csv_dir.iterdir()
        if fn.is_file() and fn.suffix == ".csv" and not fn.name.endswith("summary.csv")
    )

    if config.aris_dir and config.aris_dir.exists():
        all_aris = sorted(
            fn.name
            for fn in config.aris_dir.iterdir()
            if fn.is_file() and fn.suffix == ".aris"
        )
    else:
        all_aris = []

    clip_datetimes: list[datetime] = []
    all_have_aris = True

    for idx, fc_name in enumerate(fc_files):
        stripped = fc_name.removeprefix("FCe_").removesuffix("_ID_.txt")
        tokens = stripped.split("_")
        try:
            clip_datetimes.append(parse_datetime_from_tokens(tokens))
        except ValueError:
            print(
                f"{col.yellow}Warning: unable to parse datetime from {fc_name}{col.reset}"
            )
            clip_datetimes.append(datetime.min)

        linked_csvs = [name for name in all_csvs if stripped in name]
        linked_aris = [name for name in all_aris if stripped in name]

        if len(linked_csvs) == 1:
            csv_files[idx] = linked_csvs[0]
            if len(linked_aris) == 1:
                aris_with_csv[idx] = linked_aris[0]
        elif len(linked_csvs) > 1:
            print(
                f"{col.yellow}Warning: {fc_name} has multiple linked CSVs: {linked_csvs}{col.reset}"
            )
        else:
            print(f"{col.yellow}Warning: no CSV for {fc_name}{col.reset}")

        if len(linked_aris) == 1:
            aris_with_fc[idx] = linked_aris[0]
        else:
            all_have_aris = False
            if len(linked_aris) > 1:
                print(
                    f"{col.yellow}Warning: {fc_name} has multiple linked ARIS files: {linked_aris}{col.reset}"
                )
            elif len(linked_aris) == 0 and config.aris_dir:
                print(f"{col.yellow}Warning: no ARIS for {fc_name}{col.reset}")

    return DataIndex(
        fc_files=fc_files,
        csv_files=csv_files,
        aris_with_csv=aris_with_csv,
        aris_with_fc=aris_with_fc,
        clip_datetimes=clip_datetimes,
        all_aris=all_aris,
        all_have_aris=all_have_aris,
    )


def compare_datasets_ignore_trackid(
    data_csv: Iterable[dict], data_fc: Iterable[dict]
) -> ClipComparison:
    def record_without_trackid(record: dict) -> tuple:
        return (
            record.get("frame_id"),
            record.get("direction"),
            record.get("r_m"),
            record.get("theta"),
        )

    set_csv = {record_without_trackid(rec) for rec in data_csv}
    set_fc = {record_without_trackid(rec) for rec in data_fc}

    common = set_csv & set_fc
    missing_in_fc = set_csv - set_fc
    missing_in_csv = set_fc - set_csv

    def count_direction(records: Iterable[dict], direction: str) -> int:
        return sum(1 for rec in records if rec.get("direction") == direction)

    in_fc_upstream = count_direction(data_fc, "up")
    in_fc_downstream = count_direction(data_fc, "down")
    in_csv_upstream = count_direction(data_csv, "up")
    in_csv_downstream = count_direction(data_csv, "down")

    missing_in_fc_upstream = sum(1 for rec in missing_in_fc if rec[1] == "up")
    missing_in_fc_downstream = sum(1 for rec in missing_in_fc if rec[1] == "down")
    missing_in_csv_upstream = sum(1 for rec in missing_in_csv if rec[1] == "up")
    missing_in_csv_downstream = sum(1 for rec in missing_in_csv if rec[1] == "down")

    set_fc_len = len(set_fc) or 1

    return ClipComparison(
        file_completely_correct=len(common) == len(set_fc),
        exact_matches=len(common),
        in_fc_upstream=in_fc_upstream,
        in_fc_downstream=in_fc_downstream,
        in_csv_upstream=in_csv_upstream,
        in_csv_downstream=in_csv_downstream,
        missing_in_fc=len(missing_in_fc),
        missing_in_csv=len(missing_in_csv),
        missing_in_fc_perc=100 * len(missing_in_fc) / set_fc_len,
        missing_in_csv_perc=100 * len(missing_in_csv) / set_fc_len,
        missing_in_fc_upstream=missing_in_fc_upstream,
        missing_in_fc_downstream=missing_in_fc_downstream,
        missing_in_csv_upstream=missing_in_csv_upstream,
        missing_in_csv_downstream=missing_in_csv_downstream,
    )


def load_clip_data(index: DataIndex, config: Config) -> CombinedData:
    comparisons: list[ClipComparison] = []
    fc_records: list[dict] = []
    csv_records: list[dict] = []
    datetimes: list[datetime] = []
    missing_csv_count = 0

    for fc_name, csv_name, dt in zip(
        index.fc_files, index.csv_files, index.clip_datetimes
    ):
        if csv_name is None:
            missing_csv_count += 1
            continue

        fc_path = config.fc_dir / fc_name
        csv_path = config.csv_dir / csv_name

        data_fc, *_ = parse_fc_file(str(fc_path))
        data_csv, *_ = parse_csv_file(str(csv_path))

        comparisons.append(compare_datasets_ignore_trackid(data_csv, data_fc))
        datetimes.append(dt)
        fc_records.extend(data_fc)
        csv_records.extend(data_csv)

    if missing_csv_count:
        print(
            f"{col.yellow}Warning: {missing_csv_count} FC files missing CSV partners{col.reset}"
        )

    return CombinedData(
        comparisons=comparisons,
        fc_records=fc_records,
        csv_records=csv_records,
        datetimes=datetimes,
        missing_csv_count=missing_csv_count,
    )


def compute_coverage(index: DataIndex, config: Config) -> CoverageResult:
    if not config.aris_dir or not config.aris_dir.exists() or not index.all_aris:
        return CoverageResult(None, [], [None] * len(index.fc_files))

    try:
        from fisheye.dataloaders.didson.pyDIDSON import DIDSON  # type: ignore
    except ImportError as exc:
        print(
            f"{col.red}Warning: unable to import DIDSON ({exc}). Coverage skipped.{col.reset}"
        )
        return CoverageResult(None, [], [None] * len(index.fc_files))

    beam_width_dir = config.beam_width_dir
    if beam_width_dir and not beam_width_dir.exists():
        print(
            f"{col.yellow}Warning: beam width dir not found: {beam_width_dir}{col.reset}"
        )
        beam_width_dir = None

    all_aris_intervals: list[tuple[datetime, datetime]] = []
    analysis_intervals: list[Optional[tuple[datetime, datetime]]] = [None] * len(
        index.fc_files
    )

    for aris_name in tqdm(index.all_aris, desc="Loading ARIS coverage"):
        tokens = aris_name.removesuffix(".aris").split("_")
        try:
            start_time = parse_datetime_from_tokens(tokens)
        except ValueError:
            print(
                f"{col.yellow}Warning: unable to parse datetime from {aris_name}{col.reset}"
            )
            continue

        aris_path = config.aris_dir / aris_name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = DIDSON(
                str(aris_path),
                beam_width_dir=str(beam_width_dir) if beam_width_dir else None,
            )

        num_frames = loader.info.get("numframes", 0)
        frame_rate = loader.info.get("framerate", 0) or 1
        duration_seconds = num_frames / frame_rate
        end_time = start_time + timedelta(seconds=duration_seconds)

        interval = (start_time, end_time)
        all_aris_intervals.append(interval)

        if aris_name in index.aris_with_csv:
            slot = index.aris_with_csv.index(aris_name)
            analysis_intervals[slot] = interval

    relevant_intervals = [interval for interval in analysis_intervals if interval]
    if not relevant_intervals:
        return CoverageResult(None, all_aris_intervals, analysis_intervals)

    first_start = min(interval[0] for interval in relevant_intervals)
    last_end = max(interval[1] for interval in relevant_intervals)
    total_duration = (last_end - first_start).total_seconds()

    total_covered = sum(
        (interval[1] - interval[0]).total_seconds() for interval in relevant_intervals
    )
    coverage = (
        safe_div(total_covered, total_duration, default=None)
        if total_duration
        else None
    )

    return CoverageResult(coverage, all_aris_intervals, analysis_intervals)


def compute_aggregate_stats(
    index: DataIndex, combined: CombinedData, config: Config
) -> AggregateStats:
    comparisons = combined.comparisons

    total_crossings_csv_upstream = sum(c.in_csv_upstream for c in comparisons)
    total_crossings_csv_downstream = sum(c.in_csv_downstream for c in comparisons)
    total_crossings_fc_upstream = sum(c.in_fc_upstream for c in comparisons)
    total_crossings_fc_downstream = sum(c.in_fc_downstream for c in comparisons)

    csv_count = total_crossings_csv_upstream - total_crossings_csv_downstream
    fc_count = total_crossings_fc_upstream - total_crossings_fc_downstream
    total_counts_csv = csv_count
    total_counts_fc = fc_count
    total_counts_error = total_counts_csv - total_counts_fc
    total_counts_error_pct = safe_div(total_counts_error * 100, total_counts_fc)

    clipwise_net_fc = [c.in_fc_upstream - c.in_fc_downstream for c in comparisons]
    clipwise_net_csv = [c.in_csv_upstream - c.in_csv_downstream for c in comparisons]

    missing_in_fc_total = sum(
        c.missing_in_fc_upstream + c.missing_in_fc_downstream for c in comparisons
    )
    missing_in_csv_total = sum(
        c.missing_in_csv_upstream + c.missing_in_csv_downstream for c in comparisons
    )
    number_of_correct_matches = sum(c.exact_matches for c in comparisons)

    precision = safe_div(
        number_of_correct_matches, number_of_correct_matches + missing_in_csv_total
    )
    recall = safe_div(
        number_of_correct_matches, number_of_correct_matches + missing_in_fc_total
    )

    error_upstream = sum(
        c.missing_in_fc_upstream + c.missing_in_csv_upstream for c in comparisons
    )
    error_downstream = sum(
        c.missing_in_fc_downstream + c.missing_in_csv_downstream for c in comparisons
    )

    def accumulate_distance_counts(records: Iterable[dict]) -> tuple[int, int]:
        near_total = 0
        far_total = 0
        for record in records:
            distance = record.get("r_m")
            direction = record.get("direction")
            if distance is None or direction not in {"up", "down"}:
                continue
            delta = 1 if direction == "up" else -1
            if distance <= config.near_bank_cutoff_distance:
                near_total += delta
            else:
                far_total += delta
        return near_total, far_total

    near_fc_count, far_fc_count = accumulate_distance_counts(combined.fc_records)
    near_csv_count, far_csv_count = accumulate_distance_counts(combined.csv_records)

    if clipwise_net_fc:
        min_counts = min(clipwise_net_fc)
        max_counts = max(clipwise_net_fc)
        if clipwise_net_csv:
            min_counts = min(min_counts, min(clipwise_net_csv))
            max_counts = max(max_counts, max(clipwise_net_csv))
    else:
        min_counts = max_counts = 0

    distances = [
        record.get("r_m")
        for record in combined.fc_records + combined.csv_records
        if record.get("r_m") is not None
    ]
    if distances:
        min_distance = float(min(distances))
        max_distance = float(max(distances))
    else:
        min_distance = max_distance = 0.0

    valid_csvs = sum(1 for name in index.csv_files if name is not None)
    csvs_missing = len(index.csv_files) - valid_csvs
    valid_fcs = len(index.fc_files)
    fcs_missing = sum(1 for name in index.fc_files if name is None)
    valid_ariss = sum(1 for name in index.aris_with_fc if name is not None)
    ariss_missing = len(index.aris_with_fc) - valid_ariss

    return AggregateStats(
        csv_count=csv_count,
        fc_count=fc_count,
        total_counts_csv=total_counts_csv,
        total_counts_fc=total_counts_fc,
        total_counts_error=total_counts_error,
        total_counts_error_pct=total_counts_error_pct,
        near_csv_count=near_csv_count,
        near_fc_count=near_fc_count,
        far_csv_count=far_csv_count,
        far_fc_count=far_fc_count,
        precision=precision,
        recall=recall,
        total_crossings_csv_upstream=total_crossings_csv_upstream,
        total_crossings_fc_upstream=total_crossings_fc_upstream,
        total_crossings_csv_downstream=total_crossings_csv_downstream,
        total_crossings_fc_downstream=total_crossings_fc_downstream,
        error_upstream=error_upstream,
        error_downstream=error_downstream,
        number_of_correct_matches=number_of_correct_matches,
        missing_in_fc_total=missing_in_fc_total,
        missing_in_csv_total=missing_in_csv_total,
        clipwise_net_fc=clipwise_net_fc,
        clipwise_net_csv=clipwise_net_csv,
        min_counts=min_counts,
        max_counts=max_counts,
        min_distance=min_distance,
        max_distance=max_distance,
        valid_csvs=valid_csvs,
        csvs_missing=csvs_missing,
        valid_fcs=valid_fcs,
        fcs_missing=fcs_missing,
        valid_ariss=valid_ariss,
        ariss_missing=ariss_missing,
    )


def generate_tables(
    stats: AggregateStats, config: Config
) -> tuple[list[list], list[list], list[list], list[list]]:
    if stats.far_fc_count:
        far_error_perc_str = f"{safe_div((stats.far_csv_count - stats.far_fc_count) * 100, stats.far_fc_count):.2f}%"
    else:
        far_error_perc_str = ""

    if stats.near_fc_count:
        near_error_perc_str = f"{safe_div((stats.near_csv_count - stats.near_fc_count) * 100, stats.near_fc_count):.2f}%"
    else:
        near_error_perc_str = ""

    precision_str = f"{stats.precision:.2f}%"
    recall_str = f"{stats.recall:.2f}%"

    counts_summary = [
        ["Counts Summary"],
        ["Metric", "Predicted", "Corrected", "Error", "Error %"],
        [
            "Total Counts",
            stats.total_counts_csv,
            stats.total_counts_fc,
            stats.total_counts_error,
            f"{stats.total_counts_error_pct:.2f}%",
        ],
        [
            f"Near-range (<={config.near_bank_cutoff_distance}m) Counts",
            stats.near_csv_count,
            stats.near_fc_count,
            stats.near_csv_count - stats.near_fc_count,
            near_error_perc_str,
        ],
        [
            f"Far-range (>{config.near_bank_cutoff_distance}m) Counts",
            stats.far_csv_count,
            stats.far_fc_count,
            stats.far_csv_count - stats.far_fc_count,
            far_error_perc_str,
        ],
    ]

    total_crossings_fc = (
        stats.total_crossings_fc_upstream + stats.total_crossings_fc_downstream
    )
    total_crossings_error = stats.error_upstream + stats.error_downstream

    crossings_summary = [
        ["Crossings Summary"],
        ["Metric", "Predicted", "Corrected", "|Error|", "Error %"],
        [
            "Total Crossings",
            stats.total_crossings_csv_upstream + stats.total_crossings_csv_downstream,
            total_crossings_fc,
            total_crossings_error,
            (
                f"{safe_div(total_crossings_error * 100, total_crossings_fc):.2f}%"
                if total_crossings_fc
                else ""
            ),
        ],
        [
            "Upstream Crossings",
            stats.total_crossings_csv_upstream,
            stats.total_crossings_fc_upstream,
            stats.error_upstream,
            (
                f"{safe_div(stats.error_upstream * 100, stats.total_crossings_fc_upstream):.2f}%"
                if stats.total_crossings_fc_upstream
                else ""
            ),
        ],
        [
            "Downstream Crossings",
            stats.total_crossings_csv_downstream,
            stats.total_crossings_fc_downstream,
            stats.error_downstream,
            (
                f"{safe_div(stats.error_downstream * 100, stats.total_crossings_fc_downstream):.2f}%"
                if stats.total_crossings_fc_downstream
                else ""
            ),
        ],
    ]

    analysis_summary = [
        ["Crossings"],
        ["Correct", "False-Positives", "False-Negatives", "Precision", "Recall"],
        [
            stats.number_of_correct_matches,
            stats.missing_in_fc_total,
            stats.missing_in_csv_total,
            precision_str,
            recall_str,
        ],
    ]

    percent_missing_csv = safe_div(stats.csvs_missing * 100, stats.valid_fcs)
    percent_missing_aris = safe_div(stats.ariss_missing * 100, stats.valid_fcs)

    input_data_summary = [
        ["Input Data"],
        ["Name", "Type", "Format", "# Files", "# Missing", "% Missing"],
        [
            config.csv_label,
            "CSV",
            ".csv",
            stats.valid_csvs,
            stats.csvs_missing,
            f"{percent_missing_csv:.2f}%",
        ],
        [
            config.fc_label,
            "FCe",
            ".txt",
            stats.valid_fcs,
            stats.fcs_missing,
            "-",
        ],
        [
            "ARIS",
            "ARIS",
            ".aris",
            stats.valid_ariss,
            stats.ariss_missing,
            f"{percent_missing_aris:.2f}%",
        ],
    ]

    return counts_summary, crossings_summary, analysis_summary, input_data_summary


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_plots(
    plot_dir: Path,
    plot_filenames: dict[str, str],
    stats: AggregateStats,
    combined: CombinedData,
    coverage: CoverageResult,
    include_coverage_plot: bool,
) -> None:
    ensure_directory(plot_dir)

    if include_coverage_plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        ticks: list[int] = []
        labels: list[str] = []
        row_index = 0

        if coverage.all_aris_intervals:
            for start, end in coverage.all_aris_intervals:
                ax.barh(row_index, end - start, left=start, height=1, color="green")
            ticks.append(row_index)
            labels.append("All ARIS")
            row_index += 1

        if any(coverage.analysis_intervals):
            for interval in coverage.analysis_intervals:
                if interval:
                    start, end = interval
                    ax.barh(row_index, end - start, left=start, height=1, color="blue")
            ticks.append(row_index)
            labels.append("Included in Analysis")

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        fig.autofmt_xdate()
        ax.set_xlabel("Date-Time")
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_ylabel("")
        ax.set_title("Coverage Visualization")
        plt.tight_layout()
        plt.savefig(plot_dir / plot_filenames["coverage"])
        plt.close()

    if combined.datetimes:
        plt.figure(figsize=(10, 4))
        for dt, fc_value, csv_value in zip(
            combined.datetimes, stats.clipwise_net_fc, stats.clipwise_net_csv
        ):
            plt.plot(
                [dt, dt], [fc_value, csv_value], color="black", alpha=0.5, zorder=1
            )
            plt.plot(
                [dt, dt],
                [0, min(fc_value, csv_value)],
                color="black",
                alpha=0.15,
                zorder=1,
            )
            plt.scatter(
                dt,
                fc_value,
                color="blue",
                marker="o",
                label="Human-Corrected Counts",
                zorder=2,
            )
            plt.scatter(
                dt,
                csv_value,
                color="red",
                marker="x",
                label="Fisheye-Predicted Counts",
                zorder=3,
            )
        plt.axhline(y=0, color="black", alpha=1, zorder=0)
        ax = plt.gca()
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xticks(rotation=45, ha="right")
        plt.title("Counts per Clip")
        plt.ylabel("Count")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
    plt.savefig(plot_dir / plot_filenames["count_v_date"])
    plt.close()

    plt.figure(figsize=(8, 4))
    bins = list(range(stats.min_counts, stats.max_counts + 2)) or [0, 1]
    plt.hist(
        stats.clipwise_net_fc, bins=bins, alpha=0.5, label="Human-Corrected Counts"
    )
    if stats.clipwise_net_csv:
        plt.hist(
            stats.clipwise_net_csv,
            bins=bins,
            alpha=0.5,
            label="Fisheye-Predicted Counts",
        )
    plt.title("Histogram of Net Counts")
    plt.xlabel("Count")
    plt.xticks(bins)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / plot_filenames["count_hist"])
    plt.close()

    num_distance_bins = math.ceil(stats.max_distance) - math.floor(stats.min_distance)
    step_size = 0.5 if num_distance_bins < 10 else 1
    bins_distance = np.arange(
        math.floor(stats.min_distance), math.ceil(stats.max_distance) + 1, step_size
    )
    if bins_distance.size <= 1:
        bins_distance = np.array([stats.min_distance, stats.max_distance + step_size])

    plt.figure(figsize=(8, 4))
    plt.hist(
        [
            record.get("r_m")
            for record in combined.fc_records
            if record.get("r_m") is not None
        ],
        bins=bins_distance,
        alpha=0.5,
        label="Human-Corrected Counts",
    )
    plt.hist(
        [
            record.get("r_m")
            for record in combined.csv_records
            if record.get("r_m") is not None
        ],
        bins=bins_distance,
        alpha=0.5,
        label="Fisheye-Predicted Counts",
    )
    plt.title("Counts by Distance")
    plt.xlabel("Distance (m)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / plot_filenames["count_by_distance"])
    plt.close()


def header(c: canvas.Canvas, _doc: SimpleDocTemplate) -> None:
    username = os.getlogin()
    generated_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.saveState()
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.lightgrey)
    width, height = letter
    c.drawRightString(
        width - 40, height - 20, f"Generated by {username} at {generated_on}"
    )
    c.restoreState()


def scale_tables(
    tables: Sequence[Table], style: Optional[TableStyle] = None
) -> list[Table]:
    dummy_canvas = canvas.Canvas(BytesIO())
    table_widths = []
    col_widths_list = []
    for table in tables:
        width, _ = table.wrapOn(dummy_canvas, 1000, 1000)
        table_widths.append(width)
        widths_attr = getattr(table, "_colWidths", None)
        col_widths = widths_attr[:] if widths_attr else []
        col_widths_list.append(col_widths)

    max_width = max(table_widths)
    scaled_tables: list[Table] = []
    for table, widths in zip(tables, col_widths_list):
        natural_total = sum(widths)
        scale_factor = max_width / natural_total if natural_total else 1
        scaled_widths = [width * scale_factor for width in widths] if widths else None
        new_table = Table(table._cellvalues, colWidths=scaled_widths)
        base_style = getattr(table, "_tblStyle", None)
        if base_style is not None:
            new_table.setStyle(base_style)
        if style:
            new_table.setStyle(style)
        scaled_tables.append(new_table)

    return scaled_tables


def build_image(
    path: Path, max_width: float = 400, max_height: Optional[float] = None
) -> Image:
    reader = ImageReader(str(path))
    width, height = reader.getSize()
    if width == 0 or height == 0:
        return Image(str(path))

    aspect = height / width
    target_width = max_width
    target_height = target_width * aspect

    if max_height is not None and target_height > max_height:
        target_height = max_height
        target_width = target_height / aspect

    img = Image(str(path), width=target_width, height=target_height)
    return img


def create_pdf(
    config: Config,
    plot_filenames: dict[str, str],
    stats: AggregateStats,
    index: DataIndex,
    coverage: CoverageResult,
    coverage_included: bool,
    tables: tuple[list[list], list[list], list[list], list[list]],
) -> None:
    report_figures_dir = config.report_figures_dir
    report_fp = config.report_fp
    counts_table, crossings_table, analysis_table, input_data_table = tables

    style = TableStyle(
        [
            ("SPAN", (0, 0), (-1, 0)),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, 1), (-1, 1), 0.75, colors.grey),
            ("LINEBELOW", (0, 2), (-1, 2), 0.5, colors.grey),
            ("LINEBELOW", (0, 3), (-1, 3), 0.5, colors.grey),
            ("LINEBELOW", (0, 4), (-1, 4), 0.5, colors.grey),
            ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
        ]
    )

    table_counts = Table(counts_table)
    table_crossings = Table(crossings_table)
    table_analysis = Table(analysis_table)
    table_input = Table(input_data_table)

    for table in (table_counts, table_crossings, table_analysis, table_input):
        table.setStyle(style)

    table_counts, table_crossings, table_analysis, table_input = scale_tables(
        [table_counts, table_crossings, table_analysis, table_input], style
    )

    ensure_directory(report_figures_dir)
    doc = SimpleDocTemplate(str(report_fp), pagesize=letter)
    styles = getSampleStyleSheet()

    date_range_text = "N/A"
    if index.clip_datetimes:
        date_range_text = f"{min(index.clip_datetimes)} to {max(index.clip_datetimes)}"

    coverage_text = (
        f" with {coverage.coverage * 100:.2f}% coverage"
        if coverage.coverage is not None
        else ""
    )

    title_text = (
        "Summary Analysis: Human-Corrected Counts vs. Model-Predicted Count<br/>"
        f"{date_range_text}"
    )

    purpose_text = (
        "The function of this document is to understand how the FishEye tool performs "
        "compared to human-corrected counts to be able to quantitatively assess levels of "
        "automation and human-in-the-loop with the FishEye tool."
    )

    definitions_text = (
        "For the purpose of this document:<br/> "
        "<b>Crossings</b>: The number of fish that cross the centerline in either direction in a given clip<br/>"
        "<b>Counts</b>: The number of fish that cross the centerline upstream minus the number of fish that cross the centerline downstream in a given clip.<br/>"
        "Both crossings and counts only look at the overall movement of each, multiple crossings cancelled out. "
        "Ustream and downstream crossings do not cancel out, upstream counts can cancel out downstream counts"
    )

    input_data_text = (
        "When Fisheye is used it generates 2 files: an FCe file and a CSV file. When a human annotator opens "
        "the ARIS file and modifies the counts it overwrites the FCe file but does not change the CSV file. "
        "This document compares the corrected FCe file to the original Fisheye predictions (CSV) file to assess "
        "the accuracy of the FishEye tool."
    )

    data_summary_text = [
        (
            "This document treats the corrected files (FCe) as the canonical list of files and as the ground truth. "
            "If an FCe file is present but the CSV file is missing, this document will not include this file in the analysis."
        ),
        (
            "ARIS files are only used to calculate a coverage metric. If ARIS files are missing or a directory is "
            "not provided, this section will not be included."
        ),
        (
            "Files are drawn from: <br/>&nbsp;&nbsp; "
            f"{config.fc_label}: {config.fc_dir} <br/>&nbsp;&nbsp; {config.csv_label}: {config.csv_dir} "
            f" <br/>&nbsp;&nbsp; ARIS: {config.aris_dir if config.aris_dir else 'N/A'}"
        ),
        f"Files start times span from: {date_range_text}{coverage_text}",
    ]

    story: list = []
    story.append(Paragraph(title_text, styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Document Purpose", styles["Heading1"]))
    story.append(Paragraph(purpose_text, styles["Normal"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Data Summary", styles["Heading1"]))
    story.append(Paragraph(input_data_text, styles["Normal"]))

    bullet_list = ListFlowable(
        [ListItem(Paragraph(text, styles["Normal"])) for text in data_summary_text],
        bulletType="bullet",
    )
    story.append(bullet_list)
    story.append(Spacer(1, 10))
    story.append(table_input)
    story.append(Spacer(1, 20))

    if coverage_included:
        story.append(
            build_image(report_figures_dir / plot_filenames["coverage"], max_width=400)
        )
        story.append(Spacer(1, 20))

    story.append(Paragraph("Count Summary", styles["Heading1"]))
    story.append(Paragraph(definitions_text, styles["Normal"]))
    story.append(Spacer(1, 20))
    story.append(table_counts)
    story.append(Spacer(1, 20))
    story.append(table_crossings)
    story.append(Spacer(1, 20))
    story.append(table_analysis)
    story.append(Spacer(1, 20))
    story.append(
        build_image(report_figures_dir / plot_filenames["count_v_date"], max_width=400)
    )
    story.append(
        build_image(report_figures_dir / plot_filenames["count_hist"], max_width=400)
    )
    story.append(
        build_image(
            report_figures_dir / plot_filenames["count_by_distance"], max_width=400
        )
    )

    doc.build(story, onFirstPage=header, onLaterPages=header)
    print(f"{col.green}PDF generated: {report_fp}{col.reset}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_cli_args(argv)
    index = discover_data(config)
    coverage = (
        compute_coverage(index, config)
        if config.generate_coverage
        else CoverageResult(None, [], [None] * len(index.fc_files))
    )

    if coverage.coverage is not None:
        print(f"coverage={coverage.coverage}")
    else:
        print(f"{col.yellow}Warning: Coverage is None{col.reset}")

    combined = load_clip_data(index, config)
    stats = compute_aggregate_stats(index, combined, config)

    print(
        f"Total crossings predicted: {stats.total_crossings_csv_upstream + stats.total_crossings_csv_downstream}, "
        f"in corrected: {stats.total_crossings_fc_upstream + stats.total_crossings_fc_downstream}"
    )
    print(
        f"Total crossings FP: {stats.missing_in_fc_total} FNs, {stats.missing_in_csv_total}"
    )
    print(f"Predicted Count: {stats.csv_count}, Corrected Count: {stats.fc_count}")

    tables = generate_tables(stats, config)

    plot_filenames = {
        "count_v_date": "count_v_date.png",
        "count_hist": "count_hist.png",
        "count_by_distance": "count_by_distance.png",
        "coverage": "coverage.png",
    }

    include_coverage_plot = coverage.coverage is not None and (
        coverage.all_aris_intervals
        or any(interval for interval in coverage.analysis_intervals)
    )

    print(f"{col.blue}Generating plots...{col.reset}")
    generate_plots(
        config.report_figures_dir,
        plot_filenames,
        stats,
        combined,
        coverage,
        include_coverage_plot,
    )
    print(f"{col.blue}Generating PDF...{col.reset}")
    create_pdf(
        config, plot_filenames, stats, index, coverage, include_coverage_plot, tables
    )


if __name__ == "__main__":
    main()

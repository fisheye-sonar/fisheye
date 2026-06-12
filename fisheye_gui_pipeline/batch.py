"""Batch planning helpers for the FishEye desktop app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fisheye_gui_pipeline.config import PipelineConfig, PipelineResult

ARIS_EXTENSIONS = {".aris", ".ddf"}


def is_aris_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ARIS_EXTENSIONS


def discover_aris_files(directory: Path) -> list[Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    files = [path for path in directory.iterdir() if is_aris_file(path)]
    return sorted(files, key=lambda path: path.name.lower())


def select_directory_files(
    directory: Path,
    *,
    run_all: bool = True,
    limit: int | None = None,
) -> list[Path]:
    files = discover_aris_files(directory)
    if run_all or limit is None:
        return files
    return files[: max(0, limit)]


def _detailed_csv_paths(output_dir: Path, stem: str, frame_range_suffix: str) -> list[Path]:
    paths = []
    for path in output_dir.glob(f"*_{stem}{frame_range_suffix}.csv"):
        if path.name.endswith("_summary.csv"):
            continue
        paths.append(path)
    return sorted(paths)


def prediction_output_paths(config: PipelineConfig, input_path: Path) -> list[Path]:
    output_dir = (
        config.output_dir.expanduser().resolve()
        if config.output_dir is not None
        else input_path.expanduser().resolve().parent
    )
    paths: list[Path] = []
    stem = input_path.stem
    frame_range_suffix = config.frame_range_suffix

    if "detailed_csv" in config.export_options:
        paths.extend(_detailed_csv_paths(output_dir, stem, frame_range_suffix))
    if "fc" in config.export_options:
        paths.append(output_dir / f"FCe_{stem}{frame_range_suffix}_ID_.txt")
    if "xml" in config.export_options:
        paths.append(output_dir / f"FCe_{stem}{frame_range_suffix}_ID_.xml")
    if "mot" in config.export_options:
        paths.append(output_dir / f"{stem}{frame_range_suffix}.txt")
    return paths


def existing_output_paths(config: PipelineConfig, input_path: Path) -> list[Path]:
    return [path for path in prediction_output_paths(config, input_path) if path.is_file()]


def new_output_paths(
    config: PipelineConfig,
    input_path: Path,
    before_paths: set[Path],
) -> list[Path]:
    return sorted(set(existing_output_paths(config, input_path)) - before_paths)


def is_already_processed(config: PipelineConfig, input_path: Path) -> bool:
    expected_outputs = prediction_output_paths(config, input_path)
    if not expected_outputs:
        return False
    return all(path.is_file() for path in expected_outputs)


@dataclass
class BatchPlan:
    total: int
    to_process: list[Path]
    to_skip: list[Path]

    @property
    def process_count(self) -> int:
        return len(self.to_process)

    @property
    def skip_count(self) -> int:
        return len(self.to_skip)


def plan_batch_run(
    base_config: PipelineConfig,
    input_paths: list[Path],
    *,
    skip_already_processed: bool,
) -> BatchPlan:
    to_process: list[Path] = []
    to_skip: list[Path] = []
    for input_path in input_paths:
        file_config = base_config.with_input_path(input_path)
        if skip_already_processed and is_already_processed(file_config, input_path):
            to_skip.append(input_path)
        else:
            to_process.append(input_path)
    return BatchPlan(
        total=len(input_paths),
        to_process=to_process,
        to_skip=to_skip,
    )


@dataclass
class BatchPipelineResult:
    results: list[PipelineResult]
    skipped: list[Path]
    failures: list[tuple[Path, str]]
    output_dir: Path
    plan: BatchPlan | None = None

    @property
    def processed_count(self) -> int:
        return len(self.results)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def failed_count(self) -> int:
        return len(self.failures)

    @property
    def total_seconds(self) -> float:
        return sum(result.total_seconds for result in self.results)

    @property
    def total_upstream(self) -> int:
        return sum(result.upstream_count for result in self.results)

    @property
    def total_downstream(self) -> int:
        return sum(result.downstream_count for result in self.results)

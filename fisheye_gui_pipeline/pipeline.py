"""Configurable batch pipeline for the FishEye GUI."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from fisheye.common.system import check_disk_space, generate_job_id
from fisheye.configs import NMSConfig, ObjectDetectionConfig, YOLODatasetConfig
from fisheye.configs.factory import get_detector_config
from fisheye.configs.inference import TargetSizeConfig, TrackerConfig
from fisheye.detect.factory import DETECTOR_CLASS_REGISTRY
from fisheye.enums import DetectorType, ExportType
from fisheye.export import save_to_disk
from fisheye.pipelines import ObjectDetectionPipeline
from fisheye.pipelines.pipeline import DetectTrackCountPipeline
from fisheye_gui_pipeline.batch import (
    BatchPipelineResult,
    BatchPlan,
    existing_output_paths,
    plan_batch_run,
)
from fisheye_gui_pipeline.config import PipelineConfig, PipelineResult

logger = logging.getLogger(__name__)


def _build_pipeline(config: PipelineConfig) -> DetectTrackCountPipeline:
    detector_type = DetectorType(config.detector_type)
    checkpoint = config.checkpoint.expanduser().resolve()
    detector_cfg = get_detector_config(
        detector_type,
        weights=str(checkpoint),
        device=config.device,
    )
    detector = DETECTOR_CLASS_REGISTRY[detector_type](detector_cfg)

    target_size = TargetSizeConfig(
        min_length=config.min_target_length,
        max_length=config.max_target_length,
    )
    nms_config = NMSConfig(
        conf=config.conf,
        iou=config.iou,
        target_size=target_size,
    )
    runtime_config = ObjectDetectionConfig(
        model=detector_cfg,
        use_multithreading=config.use_multithreading,
        max_workers=config.max_workers,
        apply_nms_batchwise=True,
        apply_length_estimates_batchwise=False,
        nms_config=nms_config,
        target_size=target_size,
    )
    dataset_config = YOLODatasetConfig(
        batch_size=config.batch_size,
        workers=config.workers,
        use_blur=config.use_blur,
        use_multithreading=config.dataset_use_multithreading,
        max_workers=config.dataset_max_workers,
        img_size=config.image_size,
        start_frame=config.start_frame,
        end_frame=config.end_frame,
    )
    detector_pipeline = ObjectDetectionPipeline(model=detector, config=runtime_config)
    tracker_config = TrackerConfig(
        max_age=config.tracker_max_age,
        min_hits=config.tracker_min_hits,
        min_travel=config.tracker_min_travel,
        iou_threshold=config.tracker_iou_threshold,
        reverse=config.tracker_reverse,
    )
    return DetectTrackCountPipeline(
        detector_pipeline,
        tracker_cfg=tracker_config,
        dataset_cfg=dataset_config,
        target_size=target_size,
    )


def _count_directions(crossings: list[dict]) -> tuple[int, int]:
    upstream = sum(1 for row in crossings if row.get("Dir") == "Up")
    downstream = sum(1 for row in crossings if row.get("Dir") == "Down")
    return upstream, downstream


def run_batch_pipeline(
    base_config: PipelineConfig,
    input_paths: list[Path],
    *,
    skip_already_processed: bool = False,
    plan: BatchPlan | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_file_progress: Callable[[int, int, int, int], None] | None = None,
    on_count_progress: Callable[[int, int], None] | None = None,
) -> BatchPipelineResult:
    if not input_paths:
        raise ValueError("No ARIS files to process")

    base_config.validate_shared()

    if base_config.output_dir is None and len(input_paths) > 1:
        raise ValueError("Output directory is required when processing multiple files")

    output_dir = base_config.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    check_disk_space(output_dir)

    def progress(message: str) -> None:
        logger.info(message)
        if on_progress is not None:
            on_progress(message)

    if plan is None:
        plan = plan_batch_run(
            base_config,
            input_paths,
            skip_already_processed=skip_already_processed,
        )

    results: list[PipelineResult] = []
    failures: list[tuple[Path, str]] = []
    job_id = generate_job_id()
    total_upstream = 0
    total_downstream = 0

    def emit_file_progress() -> None:
        if on_file_progress is None:
            return
        on_file_progress(
            plan.skip_count,
            len(results),
            len(failures),
            plan.total,
        )

    def emit_count_progress() -> None:
        if on_count_progress is None:
            return
        on_count_progress(total_upstream, total_downstream)

    progress(
        f"Pre-scan: {plan.process_count} to process, {plan.skip_count} already done"
    )
    emit_file_progress()
    emit_count_progress()

    if not plan.to_process:
        return BatchPipelineResult(
            results=[],
            skipped=list(plan.to_skip),
            failures=[],
            output_dir=output_dir,
            plan=plan,
        )

    progress(f"Loading model ({base_config.checkpoint.name})…")
    pipeline = _build_pipeline(base_config)

    all_crossings: list[list[dict]] = []

    for index, input_path in enumerate(plan.to_process, start=1):
        file_config = base_config.with_input_path(input_path)
        overall = plan.skip_count + index
        progress(
            f"[{index}/{plan.process_count}] Processing {input_path.name} "
            f"(overall {overall}/{plan.total})…"
        )
        started = time.perf_counter()
        try:
            crossings = pipeline._run(
                input_path,
                file_config.resolved_output_dir,
                [
                    ExportType(option)
                    for option in file_config.export_options
                    if option != "summary_csv"
                ],
                job_id,
                file_config.upstream_direction,
                file_config.distance_offset,
                file_config.frame_range_suffix,
            )
            elapsed = time.perf_counter() - started
            all_crossings.append(crossings)
            upstream_count, downstream_count = _count_directions(crossings)
            total_upstream += upstream_count
            total_downstream += downstream_count
            results.append(
                PipelineResult(
                    input_path=input_path,
                    output_dir=file_config.resolved_output_dir,
                    total_seconds=elapsed,
                    upstream_count=upstream_count,
                    downstream_count=downstream_count,
                    exported_paths=existing_output_paths(file_config, input_path),
                )
            )
            progress(
                f"[{index}/{plan.process_count}] Finished {input_path.name} "
                f"in {elapsed:.1f}s"
            )
        except Exception as exc:
            failures.append((input_path, str(exc)))
            logger.exception("Failed on %s", input_path)
            progress(f"[{index}/{plan.process_count}] Failed {input_path.name}: {exc}")

        emit_file_progress()
        emit_count_progress()

    if "summary_csv" in base_config.export_options and all_crossings:
        progress("Writing summary CSV…")
        save_to_disk(
            all_crossings,
            str(output_dir),
            ExportType.SUMMARY_CSV,
            job_id,
            base_config.distance_offset,
            base_config.upstream_direction,
            filename_suffix=base_config.frame_range_suffix,
        )

    progress(
        f"Batch finished: {len(results)} processed, {plan.skip_count} skipped, "
        f"{len(failures)} failed"
    )
    return BatchPipelineResult(
        results=results,
        skipped=list(plan.to_skip),
        failures=failures,
        output_dir=output_dir,
        plan=plan,
    )

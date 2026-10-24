"""Clip failures preserve later work and successful summaries, with failing job status."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fisheye.endpoints.config import EndpointBatchError, EndpointPipelineError
from fisheye.enums import ExportType
from fisheye.pipelines.pipeline import DetectTrackCountPipeline
from fisheye.runner import PipelineRunner


def failing_pipeline(tmp_path, fail_names=("bad",)):
    files = [tmp_path / f"{name}.aris" for name in ("first", "bad", "last")]
    for file in files:
        file.touch()
    pipeline = object.__new__(DetectTrackCountPipeline)
    visited = []

    def run_clip(source, *args):
        visited.append(source.stem)
        if source.stem in fail_names:
            raise EndpointPipelineError(f"Failure in {source.stem}")
        return [{"file": source.name}]

    pipeline._run = run_clip
    return pipeline, files, visited


def test_failed_clip_does_not_stop_remaining_files_and_writes_manifest(tmp_path):
    pipeline, files, visited = failing_pipeline(tmp_path)
    with patch("fisheye.pipelines.pipeline.get_valid_files", return_value=files):
        with pytest.raises(EndpointBatchError) as caught:
            pipeline.run(tmp_path, tmp_path, job_id="job-1")
    error = caught.value
    assert visited == ["first", "bad", "last"]
    assert error.results == [[{"file": "first.aris"}], [{"file": "last.aris"}]]
    assert [f["file_index"] for f in error.failures] == [1]
    document = json.loads(Path(error.failure_reports[0]).read_text())
    assert document["job_id"] == "job-1"
    assert document["attempted_clips"] == 3
    assert document["successful_clips"] == 2
    assert document["failed_clips"] == 1
    assert document["failures"][0]["source_path"] == str(files[1].resolve())


def test_runner_summarizes_successes_before_raising_failed_job_status(tmp_path):
    pipeline, files, visited = failing_pipeline(tmp_path)
    runner = PipelineRunner(pipeline)
    with patch(
        "fisheye.pipelines.pipeline.get_valid_files", return_value=files
    ), patch.object(runner, "_save_summary_csv") as save:
        with pytest.raises(EndpointBatchError) as caught:
            runner.run(tmp_path, tmp_path, [ExportType.SUMMARY_CSV], "job-2", "left", 0)
    assert visited == ["first", "bad", "last"]
    assert save.call_args.args[0] == caught.value.results
    save.assert_called_once()


def test_all_failures_do_not_create_success_or_zero_count_summary(tmp_path):
    pipeline, files, visited = failing_pipeline(tmp_path, ("first", "bad", "last"))
    runner = PipelineRunner(pipeline)
    with patch(
        "fisheye.pipelines.pipeline.get_valid_files", return_value=files
    ), patch.object(runner, "_save_summary_csv") as save:
        with pytest.raises(EndpointBatchError) as caught:
            runner.run(tmp_path, tmp_path, [ExportType.SUMMARY_CSV], "job-3", "left", 0)
    assert visited == ["first", "bad", "last"]
    assert caught.value.results == []
    assert len(caught.value.failures) == 3
    save.assert_not_called()


def test_full_output_disk_preserves_successes_and_failure_details(tmp_path):
    pipeline, files, visited = failing_pipeline(tmp_path)
    with patch("fisheye.pipelines.pipeline.get_valid_files", return_value=files), patch(
        "fisheye.endpoints.processor.atomic_json", side_effect=OSError("disk full")
    ):
        with pytest.raises(EndpointBatchError) as caught:
            pipeline.run(tmp_path, tmp_path)
    assert visited == ["first", "bad", "last"]
    assert len(caught.value.results) == 2
    assert len(caught.value.failures) == 1
    assert caught.value.failure_reports == ()
    assert caught.value.report_errors[0]["message"] == "disk full"


def test_default_output_uses_failed_source_parent_and_repeated_runs_are_isolated(
    tmp_path,
):
    pipeline, files, visited = failing_pipeline(tmp_path)
    files[1] = tmp_path / "nested" / "bad.aris"
    with patch("fisheye.pipelines.pipeline.get_valid_files", return_value=files):
        with pytest.raises(EndpointBatchError) as caught:
            pipeline.run(files, None)
        assert Path(caught.value.failure_reports[0]).parent == files[1].parent
        pipeline._run = lambda source, *args: [{"file": source.name}]
        result = pipeline.run(files, None)
    assert len(result) == 3


def test_previous_non_endpoint_error_handler_retries_then_continues(tmp_path):
    files = [tmp_path / f"{name}.aris" for name in ("first", "bad", "last")]
    pipeline = object.__new__(DetectTrackCountPipeline)
    pipeline.endpoint_processor = None
    visited = []

    def run_clip(source, *args):
        visited.append(source.stem)
        if source.stem == "bad":
            raise ValueError("ordinary clip failure")
        return [{"file": source.name}]

    pipeline._run_clip = run_clip
    with patch("fisheye.pipelines.pipeline.get_valid_files", return_value=files), patch(
        "fisheye.common.generic.time.sleep"
    ):
        results = pipeline.run(files, tmp_path)
    assert visited == ["first", "bad", "bad", "bad", "last"]
    assert results == [[{"file": "first.aris"}], [], [{"file": "last.aris"}]]

"""Identity and pipeline contracts independent of trained detector weights."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from fisheye.configs.inference import TrackerConfig
from fisheye.endpoints.config import EndpointConfig, EndpointPipelineError
from fisheye.endpoints.queries import detection_records, build_queries
from fisheye.endpoints.temporal import TrackTemporalSelector
from fisheye.enums import TrackingMethod
from fisheye.track.tracker import run_tracker


def predictions():
    keys = [(0, 0), (0, 1), (1, 0)]
    boxes = np.array([[0.2, 0.3, 0.4, 0.5, 0.9], [0.6, 0.3, 0.8, 0.5, 0.8]])
    return (
        {key: boxes.copy() for key in keys},
        {key: boxes.copy() for key in keys},
        {
            keys[0]: 10,
            keys[1]: 20,
            keys[2]: 40,
        },
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_tracker_source_association_and_temporal_order(reverse):
    low, high, ids = predictions()
    tracks = run_tracker(
        low,
        high,
        10,
        10,
        TrackerConfig(reverse=reverse, min_hits=1),
        min_length=-1,
        frame_ids=ids,
        record_detection_matches=True,
    )
    assert {f.frame_num for f in tracks.frames} == {10, 20, 40}
    for frame in tracks.frames:
        assert len(frame.fish) == 2
        assert {f.bbox_index for f in frame.fish} == {0, 1}
        assert {f.det_index for f in frame.fish} == {1}
    records = detection_records(low, high, ids, 100, 80)
    queries, skipped = build_queries(
        tracks, records, "clip", TrackTemporalSelector(radius=2)
    )
    assert skipped == []
    assert len(queries) == 6
    for query in queries:
        assert query.context_frame_ids[2] == query.frame_id
        assert sorted(query.context_frame_ids) == list(query.context_frame_ids)
        assert query.bbox_xyxy_px in (
            (20.0, 24.0, 40.0, 40.0),
            (60.0, 24.0, 80.0, 40.0),
        )
    assert queries[0].context_frame_ids == (10, 10, 10, 20, 40)
    assert queries[0].context_repeated == (True, True, False, False, False)


def test_missing_association_is_skipped_instead_of_guessing():
    fish = SimpleNamespace(id=0, det_index=None, bbox_index=None)
    tracks = SimpleNamespace(frames=[SimpleNamespace(frame_num=10, fish=[fish])])
    queries, skipped = build_queries(
        tracks, {}, "clip", TrackTemporalSelector(radius=0)
    )
    assert not queries
    assert skipped[0]["reason"] == "missing_detection_association"


def test_normalized_box_conversion_requires_exact_frame_mapping():
    low, high, ids = predictions()
    records = detection_records(low, high, ids, 100, 80)
    assert records[(20, 1, 0)].bbox_xyxy_px == (20.0, 24.0, 40.0, 40.0)
    ids.pop((1, 0))
    with pytest.raises(ValueError, match="match exactly"):
        detection_records(low, high, ids, 100, 80)


def test_endpoint_config_guards_and_paths(tmp_path):
    with pytest.raises(ValueError, match="bundle"):
        EndpointConfig(enabled=True)
    cfg = EndpointConfig(
        enabled=True,
        model={"bundle": "weights/pose"},
        frame_store={"directory": "scratch"},
    )
    cfg.resolve_paths(tmp_path)
    assert cfg.model["bundle"] == str(tmp_path / "weights/pose")
    assert cfg.frame_store["directory"] == str(tmp_path / "scratch")


def test_detection_capture_handles_final_partial_batch():
    from fisheye.configs.inference import ObjectDetectionConfig
    from fisheye.pipelines.detection import ObjectDetectionPipeline
    from fisheye.endpoints.processor import EndpointSession
    from fisheye.endpoints.frame_store import BoundedFrameStore

    detector = MagicMock()
    detector.config.device = "cpu"
    detector.side_effect = [torch.zeros(2, 1, 6), torch.zeros(1, 1, 6)]
    config = ObjectDetectionConfig(
        use_multithreading=False,
        endpoints=EndpointConfig(enabled=True, model={"bundle": "dummy"}),
    )
    pipeline = ObjectDetectionPipeline(detector, config)
    pipeline.dataset = SimpleNamespace(batch_size=2)
    pipeline.metadata = SimpleNamespace()
    descriptor = {"representation": "bgs_signed_u8", "use_blur": False}

    def batch(ids):
        return (
            torch.zeros(len(ids), 3, 32, 32),
            None,
            [(None, None)] * len(ids),
            None,
            {
                "frame_ids": ids,
                "images": [np.zeros((40, 50), np.uint8) for _ in ids],
                "preprocessing": descriptor,
            },
        )

    pipeline.dataloader = [batch((5, 6)), batch((7,))]
    with BoundedFrameStore("clip") as store:
        pipeline.endpoint_session = EndpointSession(store)
        nms = MagicMock()
        nms.run.side_effect = [
            ({(0, 0): None, (0, 1): None}, {(0, 0): None, (0, 1): None}),
            ({(0, 0): None}, {(0, 0): None}),
        ]
        with patch("fisheye.pipelines.detection.NMSProcessor", return_value=nms):
            low, high, old_low, old_high = pipeline()
        assert len(low) == len(high) == 3
        assert pipeline.endpoint_session.frame_ids == {(0, 0): 5, (0, 1): 6, (1, 0): 7}
        assert store.frame_ids == (5, 6, 7)
        assert old_low == old_high == {}


def test_fail_clip_is_not_swallowed_by_legacy_retry_wrapper():
    from fisheye.pipelines.pipeline import DetectTrackCountPipeline

    pipeline = object.__new__(DetectTrackCountPipeline)
    pipeline.endpoint_processor = MagicMock()
    pipeline.endpoint_processor.session.side_effect = OSError("scratch full")
    with pytest.raises(EndpointPipelineError, match="scratch full"):
        pipeline._run("sample.aris", "/tmp")


def test_atomic_export_preserves_previous_result_when_serialization_fails(tmp_path):
    from fisheye.endpoints.processor import atomic_json

    path = tmp_path / "endpoints.json"
    atomic_json(path, {"status": "complete"})
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert path.read_text().find("complete") >= 0
    assert list(tmp_path.iterdir()) == [path]


def test_hydra_group_configures_platform_endpoint_pipeline():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from fisheye.builder import PipelineFactory
    from fisheye.configs import YOLOv5ModelConfig

    project = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(project / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "+endpoints=unblurred_signed_bgs_1f",
                "platform.inference.endpoints.device=cpu",
            ],
        )
    runtime = PipelineFactory.build_runtime_config(
        cfg.platform, project, YOLOv5ModelConfig()
    )
    assert runtime["endpoints"].enabled
    assert runtime["endpoints"].model["bundle"] == str(
        (project / "weights/endpoints/unblurred_signed_bgs_1f_seed2").resolve()
    )
    assert runtime["endpoints"].device == "cpu"


@pytest.mark.parametrize("method", [TrackingMethod.BYTETRACK, TrackingMethod.SORT])
@pytest.mark.parametrize("reverse", [False, True])
def test_frame_numbers_match_with_and_without_endpoint_associations(method, reverse):
    low, high, _ = predictions()
    ids = {key: index for index, key in enumerate(sorted(low))}
    cfg = TrackerConfig(type=method, reverse=reverse, min_hits=1)
    baseline = run_tracker(low, high, 10, 10, cfg, min_length=-1)
    endpoints = run_tracker(
        low,
        high,
        10,
        10,
        cfg,
        min_length=-1,
        frame_ids=ids,
        record_detection_matches=True,
    )
    expected = [2, 1, 0] if reverse else [0, 1, 2]
    assert [f.frame_num for f in baseline.frames] == expected
    assert [f.frame_num for f in endpoints.frames] == expected
    assert baseline.start_frame == endpoints.start_frame == 0
    assert baseline.end_frame == endpoints.end_frame == 2
    assert [[f.bbox for f in x.fish] for x in baseline.frames] == [
        [f.bbox for f in x.fish] for x in endpoints.frames
    ]


@pytest.mark.parametrize("reverse", [False, True])
def test_source_frame_ids_work_without_endpoint_associations(reverse):
    low, high, ids = predictions()
    tracks = run_tracker(
        low,
        high,
        10,
        10,
        TrackerConfig(reverse=reverse, min_hits=1),
        min_length=-1,
        frame_ids=ids,
    )
    assert [f.frame_num for f in tracks.frames] == (
        [40, 20, 10] if reverse else [10, 20, 40]
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_endpoint_toggle_preserves_crossing_frame_exports(tmp_path, reverse):
    from pathlib import Path
    from fisheye.configs import YOLODatasetConfig
    from fisheye.configs.inference import TargetSizeConfig
    from fisheye.pipelines.pipeline import DetectTrackCountPipeline

    keys = [(0, 0), (0, 1), (1, 0)]  # Last detector batch is partial.
    low = {
        key: np.array([[0.3 + 0.08 * i, 0.3, 0.5 + 0.08 * i, 0.5, 0.9]])
        for i, key in enumerate(keys)
    }
    detector = MagicMock()
    detector.endpoint_config = EndpointConfig()
    detector.apply_length_estimates_batchwise = False
    detector.return_value = (low, {k: v.copy() for k, v in low.items()}, {}, {})
    detector.metadata = SimpleNamespace(
        xdim=100, ydim=80, image_meter_width=10, image_meter_height=10
    )
    pipeline = DetectTrackCountPipeline(
        detector,
        TrackerConfig(reverse=reverse, min_hits=1),
        YOLODatasetConfig(batch_size=2),
        TargetSizeConfig(min_length=-1),
    )

    def crossing(**kwargs):
        return {
            name: kwargs[name]
            for name in ("track_id", "frame", "bbox", "crossing_direction")
        }

    with patch("fisheye.pipelines.pipeline.save_to_disk"), patch(
        "fisheye.pipelines.pipeline.format_single_crossing", side_effect=crossing
    ):
        baseline = pipeline._run_clip(Path("clip.aris"), tmp_path)
        pipeline.endpoint_processor = MagicMock()
        detector.endpoint_session.frame_ids = {key: i for i, key in enumerate(keys)}
        enabled = pipeline._run_clip(Path("clip.aris"), tmp_path)
    assert baseline == enabled
    assert len(enabled) == 1
    assert enabled[0]["frame"] == 1
    pipeline.endpoint_processor.process.assert_called_once()

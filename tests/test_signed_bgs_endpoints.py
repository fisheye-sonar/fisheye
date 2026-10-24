"""Shared native detector BGS, portable one-frame inference and JSON export."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fisheye.common.file_system import get_valid_files
from fisheye.endpoints.bundle import export_bundle, load_bundle, validate_bundle
from fisheye.endpoints.config import EndpointConfig
from fisheye.endpoints.preprocessing import CropStackPreprocessor, PreprocessingConfig
from fisheye.endpoints.processor import EndpointProcessor
from fisheye.endpoints.types import EndpointResult
from test_endpoint_capture import (
    SyntheticBaseCaptureDataset,
    SyntheticCaptureDataset,
    synthetic_config,
)


def signed_config(**kwargs):
    settings = dict(
        use_blur=False,
        only_pos_bgs=False,
        do_bg_subtract=True,
    )
    settings.update(kwargs)
    return synthetic_config(**settings)


def source_frames():
    frames = np.full((9, 10, 8), 80, np.uint8)
    frames[:, 3, 4] = np.arange(9, dtype=np.uint8) * 20
    frames[:, 6, 2] = np.arange(9, dtype=np.uint8)[::-1] * 15
    return frames


def test_signed_capture_is_exact_native_detector_channel_without_second_subtraction():
    frames = source_frames()
    ds = SyntheticBaseCaptureDataset(frames, signed_config())
    disabled = signed_config(capture_endpoint_frames=False)
    baseline = SyntheticBaseCaptureDataset(frames, disabled)
    for index in (0, 4):
        captured = ds[index]
        np.testing.assert_array_equal(captured[0], baseline[index][0])
        payload = captured[-1]
        np.testing.assert_array_equal(np.stack(payload["images"]), captured[0][..., 1])
        assert all(
            image.shape == (10, 8)
            and image.dtype == np.uint8
            and image.flags.c_contiguous
            for image in payload["images"]
        )
        assert payload["preprocessing"]["source_channel"] == 1
        assert payload["preprocessing"]["only_pos_bgs"] is False
        assert payload["preprocessing"]["background_shared_with_detector"] is True
    assert (
        ds[0][-1]["images"][0][0, 0] == 127
    )  # Neutral signed residual, not positive BGS zero.


def test_signed_capture_precedes_yolo_resizing_and_partial_batches():
    from fisheye.common.collate import yolo_collate_fn

    ds = SyntheticCaptureDataset(source_frames(), signed_config())
    batches = [yolo_collate_fn(ds[index]) for index in (0, 4)]
    assert [b[4]["frame_ids"] for b in batches] == [(2, 3, 4, 5), (6, 7)]
    assert batches[0][0].shape[-2:] != (10, 8)
    assert all(image.shape == (10, 8) for b in batches for image in b[4]["images"])
    disabled = SyntheticCaptureDataset(
        source_frames(), signed_config(capture_endpoint_frames=False)
    )
    assert torch.equal(batches[0][0], yolo_collate_fn(disabled[0])[0])


@pytest.mark.parametrize(
    "setting",
    [
        dict(use_blur=True),
        dict(only_pos_bgs=True),
        dict(do_bg_subtract=False),
        dict(return_unwarped=True),
    ],
)
def test_signed_capture_rejects_incompatible_detector_preprocessing(setting):
    with pytest.raises(ValueError, match="bgs_signed_u8 requires detector"):
        SyntheticBaseCaptureDataset(source_frames(), signed_config(**setting))


def test_signed_zero_range_uses_shared_fixed_scale_and_keeps_later_signal():
    frames = np.full((12, 10, 8), 20, np.uint8)
    frames[6:, 3, 4] = 220
    ds = SyntheticBaseCaptureDataset(
        frames,
        signed_config(
            start_frame=0, end_frame=12, batch_size=2, num_frames_bg_subtract=3
        ),
    )
    early, later = ds[0][-1], ds[6][-1]
    assert np.all(np.stack(early["images"]) == 127)
    assert later["images"][0][3, 4] == 227
    assert early["preprocessing"] == later["preprocessing"]
    assert later["preprocessing"]["normalization_policy"] == "uint8_range_fallback"


@pytest.mark.parametrize("filename", ["sample.aris", "sample.ddf"])
def test_real_sonar_signed_channel_is_native_and_detector_pixels_are_unchanged(
    filename,
):
    from fisheye.configs import YOLODatasetConfig
    from fisheye.dataloaders import create_dataloader

    config = YOLODatasetConfig(
        filepath=str(Path(__file__).parent / "data" / filename),
        start_frame=1,
        batch_size=1,
        num_frames_bg_subtract=3,
        workers=0,
        use_multithreading=False,
        use_blur=False,
        only_pos_bgs=False,
        capture_endpoint_frames=True,
    )
    loader, ds = create_dataloader(config)
    capture = next(iter(loader))
    raw, _ = ds.didson.load_frames(start_frame=1, end_frame=3)
    expected = ds._stack_preprocessed_channels(
        raw, ds.mean_blurred_frame, ds.mean_normalization_value
    )
    image = capture[4]["images"][0]
    np.testing.assert_array_equal(image, expected[0, ..., 1])
    assert image.shape == (ds.metadata.ydim, ds.metadata.xdim)
    disabled = deepcopy(config)
    disabled.capture_endpoint_frames = False
    baseline, _ = create_dataloader(disabled)
    assert torch.equal(capture[0], next(iter(baseline))[0])


class FixedBackend:
    component_id = "fixture"
    output_schema = "fixture.v1"

    def __init__(self, descriptor):
        self.input_descriptor = descriptor
        self.seen = []

    def predict(self, batch):
        self.seen.append(batch)
        return None


class FixedDecoder:
    component_id = "fixture"
    input_schema = "fixture.v1"

    def decode(self, output, batch):
        return [
            EndpointResult.from_query(
                sample.query,
                endpoints_xy_px=((2.0, 3.0), (5.0, 4.0)),
                scores={"source": "fixture"},
                transform=sample.transform,
            )
            for sample in batch.samples
        ]


def test_signed_capture_track_queries_write_global_endpoints_json_only(tmp_path):
    from fisheye.endpoints.predictor import EndpointPredictor

    pre = CropStackPreprocessor(
        PreprocessingConfig(
            representation="bgs_signed_u8", temporal_offsets=(0,), image_size=64
        )
    )
    backend = FixedBackend(pre.descriptor)
    predictor = EndpointPredictor(pre, backend, FixedDecoder())
    processor = EndpointProcessor(
        EndpointConfig(
            enabled=True,
            model={"bundle": "fixture"},
            frame_store={
                "ram_limit_mb": 0,
                "disk_limit_gb": 1,
                "directory": str(tmp_path / "scratch"),
            },
        ),
        predictor=predictor,
    )
    ds = SyntheticBaseCaptureDataset(source_frames(), signed_config())
    low, high = {}, {}
    tracks = []
    with processor.session("clip") as session:
        for batch_index, index in enumerate((0, 4)):
            payload = ds[index][-1]
            session.capture(payload, batch_index, len(payload["images"]))
            for local_index, fid in enumerate(payload["frame_ids"]):
                key = (batch_index, local_index)
                low[key] = np.array([[0.1, 0.2, 0.8, 0.7, 0.9]])
                high[key] = low[key].copy()
                tracks.append(
                    SimpleNamespace(
                        frame_num=fid,
                        fish=[SimpleNamespace(id=7, det_index=1, bbox_index=0)],
                    )
                )
        results = processor.process(
            session,
            SimpleNamespace(frames=tracks),
            low,
            high,
            SimpleNamespace(xdim=8, ydim=10, pixel_meter_size=0.01),
            tmp_path / "output",
            "clip",
        )
        assert len(results) == 6
        assert all(r.context_frame_ids == (r.frame_id,) for r in results)
        assert all(batch.images.shape[1] == 1 for batch in backend.seen)
    document = json.loads((tmp_path / "output/clip.endpoints.json").read_text())
    assert document["image_preparation"]["source_channel"] == 1
    assert document["frame_store"]["representations"].keys() == {"bgs_signed_u8"}
    assert document["predictions"][0]["endpoints_xy_px"] == [[2.0, 3.0], [5.0, 4.0]]
    assert document["predictions"][0]["track_id"] == 7
    assert sorted(p.name for p in (tmp_path / "output").iterdir()) == [
        "clip.endpoints.json"
    ]
    assert not list((tmp_path / "scratch").iterdir())


def test_signed_group_selects_seed2_shared_input_and_length_outputs():
    from hydra import compose, initialize_config_dir

    project = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(project / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="config", overrides=["+endpoints=unblurred_signed_bgs_1f"]
        )
    assert cfg.platform.dataset.use_blur is False
    assert cfg.platform.dataset.only_pos_bgs is False
    assert cfg.platform.dataset.do_bg_subtract is True
    assert cfg.platform.inference.endpoints.enabled
    assert cfg.platform.model.type == "yolov5"
    assert list(cfg.export_options) == ["summary_csv", "detailed_csv", "fc", "xml"]
    assert cfg.platform.inference.endpoints.model.bundle.endswith(
        "unblurred_signed_bgs_1f_seed2"
    )
    assert cfg.platform.inference.endpoints.track_length.edge_margin_px == 25
    assert cfg.platform.inference.endpoints.track_length.min_track_points == 5
    assert cfg.platform.inference.endpoints.track_length.percentile == 50


def test_default_endpoint_preprocessing_uses_one_signed_detector_frame():
    from fisheye.endpoints.frame_store import BoundedFrameStore
    from fisheye.endpoints.temporal import TrackTemporalSelector

    preprocessor = CropStackPreprocessor()
    context = TrackTemporalSelector().resolve((2, 3, 4), 1)
    assert context.frame_ids == (3,)
    assert preprocessor.descriptor.channels == 1
    with BoundedFrameStore("clip") as store:
        store.put(3, np.full((10, 8), 127, dtype=np.uint8))
        assert store.get(3, preprocessor.config.representation)[0, 0] == 127


def test_existing_count_output_does_not_block_endpoint_requests(tmp_path):
    source = tmp_path / "clip.aris"
    source.touch()
    old = tmp_path / "FCe_clip_ID_.txt"
    old.write_text("old count output")
    assert get_valid_files(source, tmp_path) == []
    assert get_valid_files(source, tmp_path, skip_existing_fc=False) == [source]
    assert old.read_text() == "old count output"


@pytest.fixture
def portable_signed_bundle(tmp_path):
    from ultralytics.nn.tasks import PoseModel

    torch.set_num_threads(2)
    torch.manual_seed(7)
    reference = PoseModel(
        "yolo26n-pose.yaml", ch=1, nc=1, data_kpt_shape=[2, 3], verbose=False
    ).eval()
    config = {
        "dataset": {"num_surrounding_frames": 0, "use_rgb_input": False},
        "training": {"imgsz": 64},
        "evaluation": {"candidate_confidence": 0.001},
        "experiment": {"mode": "center_pose"},
    }
    checkpoint = tmp_path / "research.pt"
    torch.save(
        {
            "model_state": reference.state_dict(),
            "model_yaml": reference.yaml,
            "config": config,
            "experiment_format": 1,
        },
        checkpoint,
    )
    profile = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/endpoints/unblurred_signed_bgs_1f_profile.json"
        ).read_text()
    )
    profile["preprocessing"]["image_size"] = 64
    bundle = tmp_path / "bundle"
    export_bundle(checkpoint, bundle, profile)
    return reference, bundle


def test_signed_single_frame_bundle_export_load_and_numerical_parity(
    portable_signed_bundle,
):
    from fisheye.endpoints.types import PreparedBatch

    reference, bundle = portable_signed_bundle
    loaded = load_bundle(bundle)
    image = torch.rand(1, 1, 64, 64)
    batch = PreparedBatch(
        image,
        torch.ones(1, 1, 64, 64, dtype=torch.bool),
        [],
        loaded.backend.input_descriptor,
    )
    with torch.inference_mode():
        expected = reference(image)[0]
    actual = loaded.backend.predict(batch).prediction[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert loaded.preprocessing["temporal_offsets"] == [0]
    spec = json.loads((bundle / "inference_spec.json").read_text())
    spec["image_preparation"]["only_positive"] = True
    (bundle / "inference_spec.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="signed/positive"):
        validate_bundle(bundle)


def test_complete_sonar_pipeline_exports_json_and_preserves_old_count_file(
    tmp_path, portable_signed_bundle
):
    from fisheye.configs import YOLODatasetConfig, YOLOv5ModelConfig
    from fisheye.configs.inference import ObjectDetectionConfig, TargetSizeConfig
    from fisheye.pipelines.detection import ObjectDetectionPipeline
    from fisheye.pipelines.pipeline import DetectTrackCountPipeline

    _, bundle = portable_signed_bundle

    class FixtureDetector(torch.nn.Module):
        """Deterministic detections exercise the real endpoint model and pipeline."""

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()
            self.config = YOLOv5ModelConfig(weights="fixture", device="cpu")

        def forward(self, image):
            batch, _, height, width = image.shape
            return image.new_tensor(
                [width / 2, height / 2, width / 3, height / 10, 0.99, 1.0]
            ).repeat(batch, 1, 1)

    detector = FixtureDetector()
    detect_pipe = ObjectDetectionPipeline(
        detector,
        ObjectDetectionConfig(
            model=detector.config,
            use_multithreading=False,
            endpoints=EndpointConfig(enabled=True, model={"bundle": str(bundle)}),
        ),
    )
    pipeline = DetectTrackCountPipeline(
        detect_pipe,
        dataset_cfg=YOLODatasetConfig(
            use_blur=False,
            only_pos_bgs=False,
            batch_size=1,
            num_frames_bg_subtract=3,
            use_multithreading=False,
        ),
        target_size=TargetSizeConfig(min_length=-1),
    )
    fixture = Path(__file__).parent / "data/sample.aris"
    output = tmp_path / "output"
    output.mkdir()
    old = output / "FCe_sample_ID_.txt"
    old.write_text("existing count result")
    results = pipeline.run(fixture, output, export_types=[])
    assert len(results) == 1
    document = json.loads((output / "sample.endpoints.json").read_text())
    assert document["counts"]["queries"] > 0
    assert sum(document["counts"]["results"].values()) == document["counts"]["queries"]
    assert document["image_preparation"]["source_channel"] == 1
    assert document["resolved_preprocessing"]["temporal_offsets"] == [0]
    assert all(len(p["context_frame_ids"]) == 1 for p in document["predictions"])
    assert sorted(p.name for p in output.iterdir()) == [
        "FCe_sample_ID_.txt",
        "sample.endpoints.json",
    ]
    assert old.read_text() == "existing count result"

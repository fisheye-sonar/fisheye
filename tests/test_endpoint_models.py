"""Endpoint decoder contracts and portable, strictly checked model loading."""

from copy import deepcopy
import json

import pytest
import torch

from fisheye.endpoints.backends import PoseOutput
from fisheye.endpoints.bundle import export_bundle, load_bundle, validate_bundle
from fisheye.endpoints.decoders import (
    NativeHeatmapDecoder,
    PoseDecoder,
    TargetHeatmapDecoder,
    native_map_to_points,
    select_detection,
    target_heatmap_distribution,
)
from fisheye.endpoints.types import (
    CoordinateTransform,
    EndpointQuery,
    InputDescriptor,
    PreparedBatch,
    PreparedSample,
)


def prepared_batch(channels=5):
    query = EndpointQuery(
        "q", "clip", 2, 10, "d", (10.0, 20.0, 40.0, 50.0), (0, 1, 2, 3, 4), (False,) * 5
    )
    transform = CoordinateTransform(
        (10.0, 20.0), (2.0, 2.0), (0, 0), (32, 32), (64, 64)
    )
    sample = PreparedSample(query, transform, torch.tensor([0.0, 0.0, 60.0, 60.0]))
    return PreparedBatch(
        torch.zeros(1, channels, 64, 64),
        torch.ones(1, 1, 64, 64, dtype=torch.bool),
        [sample],
        InputDescriptor(
            channels=channels,
            representations=("bgs_positive_u8",),
            temporal_offsets=(-2, -1, 0, 1, 2),
            fixed_hw=(64, 64),
        ),
    )


def test_pose_matching_rejects_unrelated_and_nonfinite_candidates():
    rows = torch.tensor(
        [
            [0, 0, 10, 10, 0.9, 0, 1, 1, 0.1, 9, 9, 0.2],
            [0, 0, 10, 10, 0.8, 0, 2, 2, 0.3, 8, 8, 0.4],
            [20, 20, 30, 30, 0.99, 0, 22, 22, 0.9, 28, 28, 0.9],
            [0, 0, 10, 10, float("nan"), 0, 1, 1, 0.1, 9, 9, 0.2],
        ]
    )
    assert torch.equal(
        select_detection(rows, torch.tensor([0.0, 0.0, 10.0, 10.0])), rows[0]
    )
    assert select_detection(rows, torch.tensor([100.0, 100.0, 110.0, 110.0])) is None


def test_pose_decoder_preserves_endpoint_confidence_order_and_raw_low_scores():
    pytest.importorskip("ultralytics")
    batch = prepared_batch()
    detections = torch.tensor([[[0, 0, 60, 60, 0.002, 0, 50, 20, 0.2, 10, 30, 0.8]]])
    result = PoseDecoder().decode(PoseOutput(detections, end2end=True), batch)[0]
    assert result.status == "ok"
    assert result.endpoints_xy_px == ((15.0, 35.0), (35.0, 30.0))
    assert result.scores["keypoint_confidence"] == pytest.approx([0.8, 0.2])
    assert result.scores["yolo_confidence"] == pytest.approx(0.002)
    missing = PoseDecoder().decode(PoseOutput(torch.zeros(1, 0, 12), True), batch)[0]
    assert missing.status == "no_prediction"
    assert missing.endpoints_xy_px is None and missing.length_pixels is None


def test_target_heatmap_support_and_cell_center_coordinates():
    batch = prepared_batch(channels=6)
    # Only the left output column has real-image support. Right logits must not
    # affect the distribution, even when they dominate unmasked softmax.
    batch.support[:, :, :, 32:] = False
    logits = torch.tensor(
        [[[[0.0, 100.0], [0.0, 100.0]], [[0.0, 100.0], [0.0, 100.0]]]]
    )
    log_probs, grid, valid = target_heatmap_distribution(logits, batch.support)
    assert torch.equal(valid[0, 0], torch.tensor([[True, False], [True, False]]))
    torch.testing.assert_close(
        log_probs.exp()[0, 0], torch.tensor([0.5, 0.0, 0.5, 0.0])
    )
    torch.testing.assert_close(
        (log_probs.exp() @ grid)[0, 0], torch.tensor([0.25, 0.5])
    )
    result = TargetHeatmapDecoder().decode(logits, batch)[0]
    assert result.endpoints_xy_px == ((18.0, 36.0), (18.0, 36.0))
    assert result.scores["endpoint_entropy"] == pytest.approx(
        [torch.log(torch.tensor(2.0)).item()] * 2
    )
    with pytest.raises(ValueError, match="No real-image support"):
        target_heatmap_distribution(logits, torch.zeros_like(batch.support))


def test_native_decoder_raw_power_and_degenerate_map():
    maps = torch.zeros(1, 2, 4, 5)
    maps[0, 0, 1, 3] = -2.0
    maps[0, 1, 2, 1] = 2.0
    expected = torch.tensor([[[3.0, 1.0], [1.0, 2.0]]])
    torch.testing.assert_close(
        native_map_to_points(maps, differentiable=True), expected
    )
    batch = prepared_batch()
    batch.samples[0].transform = CoordinateTransform(
        (10.0, 20.0), (1.0, 1.0), (0, 0), (64, 64), (64, 64)
    )
    result = NativeHeatmapDecoder(differentiable=True).decode(
        torch.zeros(1, 2, 64, 64), batch
    )[0]
    assert result.status == "no_prediction" and result.endpoints_xy_px is None


def reference_profile():
    return {
        "preprocessing": {
            "representation": "bgs_positive_u8",
            "temporal_offsets": [-2, -1, 0, 1, 2],
            "crop_expansion": 1.0,
            "spatial_mode": "letterbox",
            "image_size": 64,
            "stride": 32,
            "target_mask": False,
            "pad_value": 114 / 255,
            "native_canvas": "stride",
        },
        "image_preparation": {
            "id": "test_positive_bgs.v1",
            "coordinate_space": "warped_original",
            "use_blur": False,
            "only_positive": True,
            "quantization": "uint8",
            "background_statistics": "Unblurred mean and positive residual range",
            "source_value_range": [0, 255],
        },
    }


@pytest.fixture
def portable_pose(tmp_path):
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import PoseModel

    # Embedded architecture only; constructing it never requests pretrained assets.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(19)
        model = PoseModel(
            "yolo26n-pose.yaml", ch=5, nc=1, data_kpt_shape=[2, 3], verbose=False
        ).eval()
        config = {
            "dataset": {"num_surrounding_frames": 2},
            "training": {"imgsz": 64, "epochs": 1},
            "evaluation": {"candidate_confidence": 0.001},
        }
        source = tmp_path / "research.pt"
        # An arbitrary object demonstrates that export discards training state:
        # only the resulting portable payload is loaded with weights_only=True.
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_yaml": model.yaml,
                "config": config,
                "optimizer": {"unused": object()},
            },
            source,
        )
        destination = tmp_path / "portable"
        export_bundle(source, destination, reference_profile())
        yield source, destination, model
    finally:
        torch.set_num_threads(previous_threads)


def test_bundle_roundtrip_has_numerical_parity_and_clean_checkpoint(portable_pose):
    _, destination, reference = portable_pose
    loaded = load_bundle(destination)
    batch = prepared_batch()
    torch.manual_seed(7)
    batch.images = torch.rand_like(batch.images)
    with torch.inference_mode():
        expected = reference(batch.images)[0]
    actual = loaded.backend.predict(batch).prediction[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    payload = torch.load(
        destination / "weights.pt", map_location="cpu", weights_only=True
    )
    assert "optimizer" not in payload
    assert loaded.decoder.input_schema == loaded.backend.output_schema


def test_pipeline_rejects_legacy_positive_bundle(portable_pose):
    from fisheye.endpoints.config import EndpointConfig
    from fisheye.endpoints.processor import EndpointProcessor

    _, destination, _ = portable_pose
    with pytest.raises(ValueError, match="requires bgs_signed_u8"):
        EndpointProcessor(
            EndpointConfig(enabled=True, model={"bundle": str(destination)})
        )


def test_bundle_rejects_tampered_profile_and_weights(portable_pose):
    _, destination, _ = portable_pose
    spec_path = destination / "inference_spec.json"
    spec = json.loads(spec_path.read_text())
    original = deepcopy(spec)
    spec["preprocessing"]["temporal_offsets"] = [0]
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="temporal offsets"):
        validate_bundle(destination)
    spec_path.write_text(json.dumps(original))
    with (destination / "weights.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_bundle(destination)


def test_export_rejects_missing_crop_semantics_before_writing(portable_pose, tmp_path):
    source, _, _ = portable_pose
    profile = reference_profile()
    del profile["preprocessing"]["crop_expansion"]
    with pytest.raises(ValueError, match="crop_expansion"):
        export_bundle(source, tmp_path / "invalid", profile)
    assert not (tmp_path / "invalid").exists()

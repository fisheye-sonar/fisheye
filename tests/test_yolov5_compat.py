"""Exercise legacy checkpoint loading through the real YOLOv5 package."""

import pickle

import pytest
import torch

from fisheye._yolov5_compat import prepare_yolov5_loading
from fisheye.configs import YOLOv5ModelConfig
from fisheye.detect.yolov5 import YOLOv5ObjectDetectionModel
from fisheye.version import get_version_from_detector


@pytest.fixture
def legacy_checkpoint(tmp_path):
    from yolov5.models.yolo import Model

    anchors = [[10, 13, 16, 30, 33, 23]]
    model = Model(
        {
            "nc": 1,
            "depth_multiple": 1.0,
            "width_multiple": 1.0,
            "anchors": anchors,
            "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
            "head": [[[-1], 1, "Detect", [1, anchors]]],
        },
        ch=3,
        nc=1,
    ).eval()
    model.fisheye_version = "test-legacy-detector"
    path = tmp_path / "detector.pt"
    torch.save({"model": model}, path)
    return path, model


def test_legacy_checkpoint_load_and_prediction_parity(legacy_checkpoint):
    path, reference = legacy_checkpoint
    original_load = torch.load
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        expected = reference(image)[0]
    detector = YOLOv5ObjectDetectionModel(
        YOLOv5ModelConfig(weights=str(path), device="cpu")
    )
    torch.testing.assert_close(detector(image), expected, rtol=1e-5, atol=1e-5)
    assert torch.load is original_load


def test_version_loading_works_without_initializing_detector(
    legacy_checkpoint, monkeypatch
):
    from yolov5.models import experimental

    monkeypatch.setattr(experimental, "torch", torch)
    path, _ = legacy_checkpoint
    assert get_version_from_detector(str(path), "yolov5") == "test-legacy-detector"


def test_explicit_weights_only_remains_enforced(legacy_checkpoint):
    from yolov5.models import experimental

    path, _ = legacy_checkpoint
    original_load = torch.load
    prepare_yolov5_loading()
    for load in (torch.load, experimental.torch.load):
        with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
            load(path, map_location="cpu", weights_only=True)
    assert torch.load is original_load


def test_shadowed_yolov5_import_has_an_actionable_error(monkeypatch):
    import yolov5

    monkeypatch.delattr(yolov5, "load")
    with pytest.raises(ImportError, match="shadowing the installed package"):
        prepare_yolov5_loading()

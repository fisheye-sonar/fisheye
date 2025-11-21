import numpy as np
from pathlib import Path
from typing import List, Tuple

import pytest

from fisheye.dataset.builder import DatasetBuilder
from fisheye.dataset.enums import DatasetFormat


class DummyDIDSON:
    """Minimal stand‑in for fisheye.dataloaders.didson.pyDIDSON.DIDSON."""

    def __init__(self, path):
        self.info = {
            "x_meter_start": 0.0,
            "x_meter_stop": 10.0,
            "y_meter_start": 10.0,
            "y_meter_stop": 0.0,
            "xdim": 100,
            "ydim": 100,
        }


class DummyFrameExtractor:
    """Minimal stand‑in for FrameExtractor."""

    def __init__(self, extra_frames=0):
        self.extra_frames = extra_frames

    def iter_frames(self, aris_path: Path, frame_idx: int):
        class _Tensor:
            def __init__(self):
                self._arr = np.zeros((10, 10), dtype=np.uint8)

            def numpy(self):
                return self._arr

        yield _Tensor()


class DummyExporter:
    """Fake export calls instead of writing to disk."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.calls: List[Tuple[dict, str, dict]] = []

    def __call__(self, bbox_data: dict, out_ann_file: str, metadata: dict):
        self.calls.append((bbox_data, out_ann_file, metadata))


@pytest.fixture
def monkeypatch_builder_dependencies(monkeypatch, tmp_path):
    # Patch DIDSON
    from fisheye.dataset import builder as builder_mod

    monkeypatch.setattr(builder_mod, "DIDSON", DummyDIDSON)

    # Patch FrameExtractor
    monkeypatch.setattr(builder_mod, "FrameExtractor", DummyFrameExtractor)

    # Patch exporter registry to always return DummyExporter
    dummy_exporter = DummyExporter(tmp_path / "annotations")

    def _get_exporter(_dataset_format):
        # DatasetBuilder calls DATASET_EXPORTER_REGISTRY[dataset_format](annotations_dir)
        # but for simplicity we just return an already‑instantiated exporter
        return lambda annotations_dir: dummy_exporter

    class _Registry(dict):
        def __getitem__(self, key):
            return _get_exporter(key)

    monkeypatch.setattr(builder_mod, "DATASET_EXPORTER_REGISTRY", _Registry())

    def fake_find_matching_aris_xml_files(aris_dir, xml_dir):
        """Patch find_matching_aris_xml_files to return dummy paths"""
        aris_path = tmp_path / "dummy.aris"
        xml_path = tmp_path / "FCe_dummy_ID_.xml"
        # Ensure paths exist so Path operations in code don’t fail
        aris_path.write_bytes(b"dummy")
        xml_path.write_text("<root></root>")
        return [aris_path], [xml_path], [], []

    monkeypatch.setattr(
        builder_mod, "find_matching_aris_xml_files", fake_find_matching_aris_xml_files
    )

    # Patch parse_aris_xml to return a measurement with one fish and one node
    def fake_parse_aris_xml(path):
        return {
            "MarkedFishMeasurement": [
                {
                    "@FrameIndex": "5",
                    "@FishID": "42",
                    "FishMeasureNode": [
                        {"@WorldPointX": "1.0", "@WorldPointY": "2.0"},
                        {"@WorldPointX": "3.0", "@WorldPointY": "4.0"},
                    ],
                }
            ]
        }

    monkeypatch.setattr(builder_mod, "parse_aris_xml", fake_parse_aris_xml)

    def fake_get_bbox_with_padding(coords_px, padding, min_padding_px, frame_shape):
        """Patch get_bbox_with_padding to something deterministic"""
        return 10, 20, 30, 40

    from fisheye.dataset import parser as parser_mod

    monkeypatch.setattr(parser_mod, "get_bbox_with_padding", fake_get_bbox_with_padding)

    return dummy_exporter


def test_dataset_builder_build_all(monkeypatch_builder_dependencies, tmp_path):
    """Test DatasetBuilder.build_all() with a dummy dataset."""
    dummy_exporter = monkeypatch_builder_dependencies

    builder = DatasetBuilder(
        aris_dir=tmp_path / "aris",
        xml_dir=tmp_path / "xml",
        out_dir=tmp_path / "out",
        dataset_format=DatasetFormat.YOLO,
    )

    builder.build_all()

    assert len(dummy_exporter.calls) == 1

    bbox_data, out_ann_file, metadata = dummy_exporter.calls[0]

    assert out_ann_file.startswith("dummy_")
    assert out_ann_file.endswith("000005")  # frame index 5, zero‑padded
    assert bbox_data["frame_idx"] == 5
    assert bbox_data["fish_id"] == 42
    assert "bbox_xy_xy" in bbox_data
    assert "fish_coords_xyxy" in bbox_data
    assert "fish_coords_meters" in bbox_data
    assert "xdim" in metadata
    assert "ydim" in metadata

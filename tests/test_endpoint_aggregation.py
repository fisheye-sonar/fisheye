"""Track-length acceptance and export contracts, independent of model quality."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fisheye.endpoints.aggregation import aggregate_track_lengths
from fisheye.endpoints.config import EndpointConfig, TrackLengthConfig
from fisheye.endpoints.cone import SonarConeGeometry
from fisheye.endpoints.processor import EndpointProcessor
from fisheye.endpoints.types import EndpointResult


def prediction(frame=0, track=1, length=10.0, points=None, status="ok"):
    return EndpointResult(
        query_id=f"clip:{track}:{frame}",
        clip_id="clip",
        frame_id=frame,
        track_id=track,
        detection_id=str(frame),
        bbox_xyxy_px=(40, 50, 200, 150),
        status=status,
        endpoints_xy_px=points or ((100.0, 100.0), (100.0 + length, 100.0)),
        length_cm=length,
        length_pixels=length,
    )


def aggregate(results, **settings):
    return aggregate_track_lengths(
        results,
        (200, 300),
        TrackLengthConfig(enabled=True, **settings),
        cone=cone_geometry(),
    )


def cone_metadata():
    return SimpleNamespace(
        beam_width_data={"beam_left": [-30.0, 0.0], "beam_right": [0.0, 30.0]},
        pixel_meter_size=0.01,
        x_meter_start=-1.5,
        y_meter_start=3.0,
    )


def cone_geometry():
    return SonarConeGeometry.from_metadata(cone_metadata())


def side_point(side, distance):
    cone = cone_geometry()
    angle = np.deg2rad(cone.angles_deg[side])
    # A point on the side at y=100, displaced inward PERPENDICULARLY.
    boundary = np.array([150 + 200 * np.tan(angle), 100.0])
    return tuple(boundary + distance * np.array(cone.inward_normals[side]))


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("endpoint", [0, 1])
def test_either_endpoint_exactly_25px_from_either_cone_side_is_discarded(
    side, endpoint
):
    points = [(150.0, 100.0), (160.0, 100.0)]
    points[endpoint] = side_point(side, 25.0)
    result = prediction(points=tuple(points))
    output = aggregate([result], min_track_points=1)
    assert output.length_outputs == {}
    assert output.prediction_filters[0]["reason"] == "endpoint_near_cone_side"
    assert output.prediction_filters[0]["nearest_cone_side_px"] == pytest.approx(25.0)
    assert result.status == "ok"


@pytest.mark.parametrize(
    "margin,distance,accepted",
    [
        (25, 25.001, True),
        (25, 24.999, False),
        (5, 6, True),
        (0, 0, False),
        (0, 0.1, True),
        (0, -30, False),
    ],
)
def test_cone_margin_is_perpendicular_tunable_and_rejects_exterior(
    margin, distance, accepted
):
    output = aggregate(
        [
            prediction(points=(side_point(0, distance), (150.0, 100.0))),
        ],
        min_track_points=1,
        edge_margin_px=margin,
    )
    assert bool(output.length_outputs) is accepted
    if distance == -30:
        assert output.prediction_filters[0]["reason"] == "endpoints_outside_sonar_cone"


@pytest.mark.parametrize(
    "points",
    [
        ((150.0, 0.0), (160.0, 0.0)),
        ((150.0, 199.0), (160.0, 199.0)),
        ((10.0, 0.0), (30.0, 0.0)),
    ],
)
def test_canvas_edges_do_not_apply_the_25px_margin(points):
    assert aggregate([prediction(points=points)], min_track_points=1).length_outputs


def test_cone_side_filter_applies_far_from_rectangular_edges():
    result = prediction(points=((85.0, 150.0), (150.0, 150.0)))
    # Every canvas edge is >=49px away, but this point is beside the cone side.
    out = aggregate([result], min_track_points=1)
    assert out.prediction_filters[0]["reason"] == "endpoint_near_cone_side"


@pytest.mark.parametrize(
    "points",
    [
        ((-1.0, 0.0), (150.0, 100.0)),
        ((150.0, 200.0), (150.0, 100.0)),
    ],
)
def test_endpoints_outside_the_image_are_still_invalid(points):
    out = aggregate([prediction(points=points)], min_track_points=1)
    assert out.prediction_filters[0]["reason"] == "endpoints_outside_frame"


@pytest.mark.parametrize(
    "change",
    [
        {"beam_width_data": None},
        {"beam_width_data": {"beam_left": [], "beam_right": []}},
        {"pixel_meter_size": 0},
        {"x_meter_start": float("nan")},
        {"beam_width_data": {"beam_left": [10.0], "beam_right": [-10.0]}},
    ],
)
def test_missing_or_invalid_cone_calibration_never_falls_back_to_image_edges(change):
    metadata = cone_metadata()
    for name, value in change.items():
        setattr(metadata, name, value)
    with pytest.raises(ValueError, match="[Ss]onar cone"):
        SonarConeGeometry.from_metadata(metadata)


def test_minimum_is_applied_after_filters_and_exactly_five_are_enough():
    results = [prediction(i, length=10.0 + i) for i in range(5)]
    results += [prediction(5, points=((1.0, 100.0), (60.0, 100.0)))]
    output = aggregate(results)
    assert output.tracks[0]["usable_points"] == 5
    assert output.tracks[0]["prediction_count"] == 6
    assert output.tracks[0]["length_cm"] == 12
    assert output.length_outputs[1]["filtered_lengths_cm"] == 12
    assert sum(p["used_for_track_length"] for p in output.prediction_filters) == 5
    short = aggregate(results[1:])
    assert not short.length_outputs
    assert short.tracks[0]["reason"] == "insufficient_usable_points"
    assert short.tracks[0]["length_cm"] is None
    assert not any(p["used_for_track_length"] for p in short.prediction_filters)
    assert aggregate(results[1:], min_track_points=4).length_outputs


@pytest.mark.parametrize(
    "percentile,expected", [(0, 10), (50, 25), (75, 32.5), (100, 40)]
)
def test_percentile_uses_all_filtered_lengths_with_linear_interpolation(
    percentile, expected
):
    results = [prediction(i, length=n) for i, n in enumerate([10.0, 20.0, 30.0, 40.0])]
    results += [prediction(10, points=((0.0, 100.0), (299.0, 100.0)), length=299)]
    out = aggregate(results[::-1], min_track_points=4, percentile=percentile)
    assert out.tracks[0]["length_cm"] == expected
    assert out.tracks[0]["length_pixels"] == expected
    deciles = {str(p): 10 + 0.3 * p for p in range(0, 101, 10)}
    assert out.tracks[0]["length_percentiles_cm"] == pytest.approx(deciles)
    assert out.length_outputs[1]["length_percentiles_cm"] == pytest.approx(deciles)
    representative = out.tracks[0]["representative"]
    assert representative["length_cm"] in [10, 20, 30, 40]
    if percentile == 50:
        assert representative["frame_id"] == 1  # Deterministic chronological tie.
        assert representative["length_cm"] == 20  # Distinct from interpolated 25.


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"status": "no_prediction"}, "no_prediction"),
        ({"length_cm": None}, "missing_physical_scale"),
        ({"length_cm": float("nan")}, "invalid_length"),
        ({"length_cm": 0}, "invalid_length"),
        ({"length_pixels": float("inf")}, "invalid_length"),
        ({"endpoints_xy_px": ((float("nan"), 40), (60, 80))}, "invalid_endpoints"),
        ({"track_id": None}, "missing_track_id"),
    ],
)
def test_unavailable_or_invalid_measurements_do_not_enter_percentile(change, reason):
    out = aggregate([replace(prediction(), **change)], min_track_points=1)
    assert not out.length_outputs
    assert out.prediction_filters[0]["reason"] == reason


def test_empty_and_unassociated_tracks_do_not_receive_lengths():
    out = aggregate_track_lengths(
        [], (200, 300), TrackLengthConfig(), cone=cone_geometry(), track_ids=[1]
    )
    assert out.tracks[0]["usable_points"] == 0
    assert out.tracks[0]["status"] == "discarded"
    assert aggregate([]).tracks == []
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([prediction(), prediction()])


@pytest.mark.parametrize(
    "settings",
    [
        {"edge_margin_px": -1},
        {"edge_margin_px": float("inf")},
        {"edge_margin_px": True},
        {"percentile": -1},
        {"percentile": 101},
        {"percentile": float("nan")},
        {"percentile": False},
        {"min_track_points": 0},
        {"min_track_points": 2.5},
        {"min_track_points": True},
        {"enabled": "false"},
    ],
)
def test_invalid_configuration_fails_before_processing(settings):
    with pytest.raises(ValueError, match="track_length"):
        EndpointConfig(track_length=settings)


def test_hydra_overrides_and_runtime_config(tmp_path):
    from hydra import compose, initialize_config_dir
    from fisheye.builder import PipelineFactory
    from fisheye.configs import YOLOv5ModelConfig

    project = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(project / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "+endpoints=unblurred_signed_bgs_1f",
                "platform.inference.endpoints.track_length.edge_margin_px=12",
                "platform.inference.endpoints.track_length.min_track_points=7",
                "platform.inference.endpoints.track_length.percentile=75",
            ],
        )
    runtime = PipelineFactory.build_runtime_config(
        cfg.platform, project, YOLOv5ModelConfig()
    )
    settings = runtime["endpoints"].track_length
    assert (
        settings.edge_margin_px,
        settings.min_track_points,
        settings.percentile,
    ) == (12, 7, 75)


@pytest.mark.parametrize("min_points", [5, 8])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("percentile,length_cm,frame", [(50, 45, 3), (75, 57.5, 5)])
def test_pipeline_filters_aggregates_and_exports_tracks(
    tmp_path, min_points, reverse, percentile, length_cm, frame
):
    from fisheye.configs import YOLODatasetConfig
    from fisheye.configs.inference import TrackerConfig, TargetSizeConfig
    from fisheye.endpoints.preprocessing import PreprocessingConfig
    from fisheye.pipelines.pipeline import DetectTrackCountPipeline
    from fisheye.export import DetailedCSVExporter, FCExporter
    from fisheye.export.inference import XMLExporter
    from fisheye.export.schema import FC_SCHEMA
    from test_exporters import sample_data
    import pandas as pd
    import xml.etree.ElementTree as ET

    # Seven observations cross the counting line; one edge result is excluded.
    detector = MagicMock()
    detector.endpoint_config = EndpointConfig()
    detector.endpoint_detection_provenance.return_value = {"test": "fixed boxes"}
    low = {
        (0, i): np.array([[0.3 + 0.05 * i, 0.3, 0.5 + 0.05 * i, 0.5, 0.9]])
        for i in range(7)
    }
    detector.return_value = (low, low, {}, {})
    metadata = sample_data()[0][0]["metadata"]
    metadata.xdim, metadata.ydim = 300, 200
    metadata.beam_width_data = pd.DataFrame(
        {
            **cone_metadata().beam_width_data,
            "beam_num": [0, 1],
            "beam_center": [-15.0, 15.0],
        }
    )
    metadata.x_meter_start = -150 * metadata.pixel_meter_size
    metadata.x_meter_stop = 150 * metadata.pixel_meter_size
    metadata.y_meter_start = 300 * metadata.pixel_meter_size
    metadata.y_meter_stop = 100 * metadata.pixel_meter_size
    detector.metadata = metadata
    processor = EndpointProcessor(
        EndpointConfig(
            track_length={
                "enabled": True,
                "min_track_points": min_points,
                "percentile": percentile,
            }
        ),
        predictor=SimpleNamespace(
            preprocessor=SimpleNamespace(
                config=PreprocessingConfig(temporal_offsets=(0,))
            ),
            provenance=lambda: {"test": "deterministic endpoints"},
        ),
    )

    def predict(store, queries, pixel_meter_size):
        return [
            EndpointResult.from_query(
                q,
                endpoints_xy_px=(
                    ((100.0, 100.0), (100.0 + (q.frame_id + 1) * 10, 100.0))
                    if q.frame_id
                    else ((1.0, 100.0), (11.0, 100.0))
                ),
                length_cm=float((q.frame_id + 1) * 10),
                length_pixels=float((q.frame_id + 1) * 10),
            )
            for q in queries
        ]

    processor.predictor.predict = predict
    pipeline = DetectTrackCountPipeline(
        detector,
        TrackerConfig(reverse=reverse, min_hits=1),
        YOLODatasetConfig(batch_size=7),
        TargetSizeConfig(min_length=-1),
    )
    pipeline.endpoint_processor = processor
    with processor.session("clip") as session:
        session.capture(
            {
                "frame_ids": tuple(range(7)),
                "images": [np.zeros((200, 300), np.uint8)] * 7,
                "preprocessing": {"representation": "bgs_signed_u8"},
            },
            0,
            7,
        )
        detector.endpoint_session = session
        with patch("fisheye.pipelines.pipeline.save_to_disk"):
            crossings = pipeline._run_clip(Path("clip.aris"), tmp_path)
    document = json.loads((tmp_path / "clip.endpoints.json").read_text())
    assert document["track_lengths"][0]["usable_points"] == 6
    assert (
        document["predictions"][0]["length_filter"]["reason"]
        == "endpoints_outside_sonar_cone"
    )
    assert len(crossings) == 1
    crossings[0]["Comment"] = "reviewer note"
    FCExporter(tmp_path).export([crossings])
    fc_row = next(tmp_path.glob("FCe_*.txt")).read_text().splitlines()[4]
    length_start = sum(
        FC_SCHEMA[name].width
        for name in list(FC_SCHEMA)[: list(FC_SCHEMA).index("L(cm)")]
    )
    comment_start = sum(
        col.width for name, col in FC_SCHEMA.items() if name != "Comment"
    )
    comment = fc_row[comment_start:]
    assert comment.startswith(" ")  # Notes must not run into the N column.
    assert "reviewer note" in comment
    if min_points == 5:
        assert crossings[0]["L(cm)"] == length_cm
        assert document["track_lengths"][0]["length_cm"] == length_cm
        assert crossings[0]["Frame#"] == frame
        assert float(fc_row[length_start : length_start + 8]) == length_cm
        assert f"length_percentile={percentile};" in comment
        deciles = json.loads(comment.split("length_percentiles_cm=", 1)[1])
        # Usable lengths are 20..70cm; the rejected 10cm observation is excluded.
        assert deciles == {str(p): 20 + 0.5 * p for p in range(0, 101, 10)}
        assert document["track_lengths"][0]["length_percentiles_cm"] == deciles
        DetailedCSVExporter(tmp_path).export([crossings])
        XMLExporter(tmp_path, upstream_direction="left").export([crossings])
        csv = next(tmp_path.glob("*.csv"))
        assert pd.read_csv(csv)["L(cm)"].iloc[0] == length_cm
        xml = ET.parse(next(tmp_path.glob("*.xml")))
        assert (
            float(xml.findall(".//FishMeasureNode")[-1].attrib["Length"]) == length_cm
        )
    else:
        assert crossings[0]["L(cm)"] == 0  # Existing external unavailable convention.
        assert document["track_lengths"][0]["length_cm"] is None
        assert document["track_lengths"][0]["length_percentiles_cm"] is None
        assert float(fc_row[length_start : length_start + 8]) == 0
        assert "length_percentile" not in comment
        assert processor.last_length_outputs == {}


def test_representative_frame_zero_is_preserved():
    from fisheye.format import format_single_crossing

    row = format_single_crossing(
        "clip.aris",
        None,
        track_id=1,
        frame=8,
        len_outputs={
            1: {
                "frame_id_closest_to_mean": 0,
                "filtered_lengths_cm": 10,
            }
        },
    )
    assert row["Frame#"] == 0


@pytest.mark.parametrize("filename", ["sample.aris", "sample.ddf"])
def test_calibrated_sides_match_real_sonar_warp_boundaries(filename):
    from fisheye.dataloaders.didson.pyDIDSON import DIDSON

    ds = DIDSON(Path(__file__).parent / "data" / filename)
    geometry = SonarConeGeometry.from_metadata(SimpleNamespace(**ds.info))
    # The raster's first interior pixels sit within ~one pixel of the continuous
    # cone side; test multiple ranges where neither range arc clips the fan.
    for y in np.linspace(0.2 * ds.info["ydim"], 0.8 * ds.info["ydim"], 4, dtype=int):
        x = ds.write_cols[ds.write_rows == y]
        points = np.array([[x.min(), y], [x.max(), y]])
        distances = (points - geometry.apex_xy_px) @ np.array(geometry.inward_normals).T
        assert np.all(distances.min(axis=1) >= -0.01)
        assert np.all(distances.min(axis=1) < 1.5)


def test_calibration_uses_outer_beam_edges_including_asymmetric_cones():
    metadata = cone_metadata()
    metadata.beam_width_data = {
        "beam_left": [-12.0, -5.0, 7.0],
        "beam_right": [-5.0, 7.0, 22.0],
        "beam_center": [-8.5, 1.0, 14.5],
    }
    geometry = SonarConeGeometry.from_metadata(metadata)
    assert geometry.angles_deg == (-12.0, 22.0)
    for side, angle in enumerate(geometry.angles_deg):
        a = np.deg2rad(angle)
        boundary = np.array(geometry.apex_xy_px) + 1000 * np.array(
            [np.sin(a), -np.cos(a)]
        )
        point = boundary + 25 * np.array(geometry.inward_normals[side])
        assert geometry.nearest_side_distance_px([point]) == pytest.approx(25.0)

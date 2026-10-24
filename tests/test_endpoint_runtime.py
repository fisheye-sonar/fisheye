from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fisheye.endpoints.frame_store import BoundedFrameStore, RepresentationDescriptor
from fisheye.endpoints.predictor import EndpointPredictor
from fisheye.endpoints.preprocessing import (
    ChannelTemporalStacker,
    CropStackPreprocessor,
    PreprocessingConfig,
    expanded_crop,
)
from fisheye.endpoints.temporal import TrackTemporalSelector
from fisheye.endpoints.types import (
    CoordinateTransform,
    EndpointQuery,
    EndpointResult,
    SpatialPreparedImage,
)


def query(frame_id=2, context=(0, 1, 2, 3, 4), **changes):
    values = dict(
        query_id="q",
        clip_id="clip",
        frame_id=frame_id,
        track_id=7,
        detection_id="det",
        bbox_xyxy_px=(4.25, 3.25, 8.5, 6.25),
        context_frame_ids=context,
        context_repeated=(False,) * len(context),
    )
    values.update(changes)
    return EndpointQuery(**values)


@pytest.fixture
def store(tmp_path):
    with BoundedFrameStore("clip", directory=tmp_path) as frames:
        for i in range(5):
            frames.put(i, np.full((13, 21), i * 50, dtype=np.uint8))
        yield frames


@pytest.mark.parametrize("radius", [0, 1, 2, 4])
def test_temporal_context_is_flexible_and_sparse(radius):
    selector = TrackTemporalSelector(radius=radius)
    frames = (5, 10, 90)
    resolved = selector.resolve(frames, 0)
    assert resolved.frame_ids == tuple(
        frames[min(max(i, 0), 2)] for i in range(-radius, radius + 1)
    )
    assert resolved.repeated == tuple(
        i < 0 or i > 2 for i in range(-radius, radius + 1)
    )
    assert len(resolved.frame_ids) == 2 * radius + 1


def test_temporal_source_offsets_leave_interior_holes_missing():
    selected = TrackTemporalSelector(offsets=(-3, 0, 1), basis="source_frames").resolve(
        (4, 9, 15), 1
    )
    assert selected.frame_ids == (6, 9, 10)
    assert selected.repeated == (False, False, False)
    with pytest.raises(ValueError, match="chronological"):
        TrackTemporalSelector().resolve((15, 9, 4), 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": -1},
        {"radius": 1.5},
        {"radius": 1, "offsets": [-2, 0, 2]},
        {"offsets": [0, 0]},
        {"offsets": [1]},
        {"offsets": [False]},
    ],
)
def test_invalid_temporal_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        TrackTemporalSelector(**kwargs)


@pytest.mark.parametrize("count", [1, 3, 5, 9])
def test_stack_preserves_temporal_order_and_original_crop(store, count):
    radius = count // 2
    offsets = tuple(range(-radius, radius + 1))
    resolved = TrackTemporalSelector(radius=radius).resolve((0, 1, 2, 3, 4), 2)
    preprocessor = CropStackPreprocessor(
        PreprocessingConfig(
            temporal_offsets=offsets,
            spatial_mode="native",
            stride=1,
            native_canvas="none",
        )
    )
    q = query(
        context=resolved.frame_ids,
        context_repeated=resolved.repeated,
        crop_xyxy_px=(2, 1, 12, 10),
    )
    batch = next(preprocessor.prepare(store, [q]))
    assert batch.images.shape == (1, count, 9, 10)
    assert torch.equal(
        batch.images[0, :, 0, 0], torch.tensor(resolved.frame_ids).float() * 50 / 255
    )
    assert batch.support.all()
    assert batch.samples[0].transform.crop_xyxy_px == (2, 1, 12, 10)


def test_crop_expands_integer_outward_bounds_before_padding():
    assert expanded_crop((10.8, 20.2, 15.1, 23.4), (100, 100)) == (4, 16, 22, 28)
    assert expanded_crop((0.2, 0.2, 1.3, 2.1), (5, 5)) == (0, 0, 4, 5)


def test_letterbox_tensor_and_inverse_geometry_match_reference(store):
    preprocessor = CropStackPreprocessor(
        PreprocessingConfig(temporal_offsets=(-2, -1, 0, 1, 2), image_size=32, stride=1)
    )
    q = query(crop_xyxy_px=(2, 1, 13, 8))
    batch = next(preprocessor.prepare(store, [q]))
    crop = torch.stack([torch.full((7, 11), i * 50 / 255) for i in range(5)])
    scale = min(32 / 7, 32 / 11)
    nh, nw = round(7 * scale), round(11 * scale)
    top, left = (32 - nh) // 2, (32 - nw) // 2
    expected = F.pad(
        F.interpolate(crop[None], (nh, nw), mode="bilinear", align_corners=False),
        (left, 32 - nw - left, top, 32 - nh - top),
        value=114 / 255,
    )
    assert torch.equal(batch.images, expected)
    transform = batch.samples[0].transform
    points = np.array([[2.75, 3.0], [8.0, 7.0]])
    np.testing.assert_allclose(
        transform.to_full(transform.to_model(points)), points, atol=1e-6
    )
    assert transform.gain_xy == (nw / 11, nh / 7)


def test_coordinate_transform_handles_integer_tensors():
    transform = CoordinateTransform((2, 3), (1.5, 2.5), (4, 5), (8, 8), (32, 32))
    points = torch.tensor([[4, 5]])
    assert torch.allclose(transform.to_full(transform.to_model(points)), points.float())


def test_target_mask_uses_nearest_and_zero_padding(store):
    q = query(crop_xyxy_px=(2, 1, 13, 8))
    batch = next(
        CropStackPreprocessor(
            PreprocessingConfig(
                temporal_offsets=(-2, -1, 0, 1, 2),
                target_mask=True,
                image_size=32,
                stride=1,
            )
        ).prepare(store, [q])
    )
    transform = batch.samples[0].transform
    local = torch.zeros(1, 7, 11)
    local[:, 2:6, 2:7] = 1
    nh, nw = round(7 * transform.gain_xy[1]), round(11 * transform.gain_xy[0])
    expected = torch.zeros(1, 32, 32)
    left, top = transform.pad_xy
    expected[:, top : top + nh, left : left + nw] = F.interpolate(
        local[None], (nh, nw), mode="nearest"
    )[0]
    assert torch.equal(batch.images[0, -1:], expected)
    assert not batch.images[0, -1][~batch.support[0, 0]].any()


def test_conditioning_mask_preserves_reference_float32_roundtrip_at_integer_edge(store):
    # In the trained implementation, 7 * float32(64/12) / float32(64/12)
    # is 7.0000004768. Its ceil includes local pixel 7 before nearest resize.
    # Removing that roundtrip shifts the conditioning mask and model outputs.
    q = query(bbox_xyxy_px=(2.0, 2.0, 7.0, 7.0), crop_xyxy_px=(0, 0, 12, 9))
    batch = next(
        CropStackPreprocessor(
            PreprocessingConfig(
                temporal_offsets=(-2, -1, 0, 1, 2),
                target_mask=True,
                image_size=64,
                stride=1,
            )
        ).prepare(store, [q])
    )
    assert batch.images[0, -1, 20, 38] == 1


def test_target_box_clips_for_model_matching_but_preserves_original_query(store):
    q = query(bbox_xyxy_px=(-2.0, -1.0, 7.0, 8.0), crop_xyxy_px=(0, 0, 12, 9))
    prep = CropStackPreprocessor(
        PreprocessingConfig(
            temporal_offsets=(-2, -1, 0, 1, 2),
            spatial_mode="native",
            native_canvas="none",
            stride=1,
        )
    )
    batch = next(prep.prepare(store, [q]))
    assert batch.samples[0].target_box.tolist() == [0.0, 0.0, 7.0, 8.0]
    assert batch.samples[0].query.bbox_xyxy_px == (-2.0, -1.0, 7.0, 8.0)
    missed = replace(q, crop_xyxy_px=(9, 9, 12, 12))
    result = EndpointPredictor(
        prep, StubBackend(prep.descriptor), StubDecoder()
    ).predict(store, [missed])[0]
    assert result.status == "invalid_query"
    assert result.reason["code"] == "box_outside_crop"


def test_native_batch_max_keeps_inverse_origin_and_support(store):
    preprocessor = CropStackPreprocessor(
        PreprocessingConfig(
            temporal_offsets=(-2, -1, 0, 1, 2),
            spatial_mode="native",
            stride=1,
            native_canvas="batch_max",
            pad_value=0,
            batch_size=2,
        )
    )
    a = query(crop_xyxy_px=(2, 1, 13, 8))
    b = replace(a, query_id="b", crop_xyxy_px=(1, 1, 9, 13))
    batch = next(preprocessor.prepare(store, [a, b]))
    assert batch.images.shape == (2, 5, 12, 11)
    assert batch.samples[0].transform.canvas_hw == (12, 11)
    assert batch.samples[0].transform.crop_hw == (7, 11)
    assert batch.support[0].sum() == 7 * 11
    assert batch.support[1].sum() == 12 * 8
    assert not batch.images[0, :, 7:].any()


def test_store_spills_losslessly_and_cleans_up(tmp_path):
    store = BoundedFrameStore(
        "clip", ram_limit_bytes=8, disk_limit_bytes=2048, directory=tmp_path
    )
    scratch = store.directory
    a = np.arange(8, dtype=np.uint8).reshape(2, 4)
    store.put(0, a)
    a[:] = 0
    store.put(1, np.full((2, 4), 255, dtype=np.uint8))
    assert store.ram_bytes == 8
    assert store.disk_bytes > 8
    assert store.peak_ram_bytes <= 8
    assert isinstance(store.get(0), np.memmap)
    np.testing.assert_array_equal(store.get(0), np.arange(8).reshape(2, 4))
    with pytest.raises(ValueError):
        store.get(1).setflags(write=True)
    store.close()
    assert not scratch.exists()
    assert store.closed
    with pytest.raises(RuntimeError, match="closed"):
        store.get(0)


def test_store_cleanup_on_exception_and_disk_capacity_failure(tmp_path):
    with pytest.raises(OSError, match="disk limit"):
        with BoundedFrameStore(
            "clip", ram_limit_bytes=0, disk_limit_bytes=10, directory=tmp_path
        ) as store:
            store.put(0, np.zeros((2, 2), dtype=np.uint8))
    assert not list(tmp_path.iterdir())


def test_named_representations_are_explicit(tmp_path):
    with BoundedFrameStore(
        "clip",
        directory=tmp_path,
        representations=(RepresentationDescriptor("raw_u8"),),
    ) as store:
        store.put(0, np.zeros((2, 4), dtype=np.uint8), "raw_u8")
        with pytest.raises(KeyError):
            store.get(0, "bgs_signed_u8")
        with pytest.raises(ValueError, match="Undeclared"):
            store.put(1, np.zeros((2, 4), dtype=np.uint8))
        with pytest.raises(ValueError, match="dimensions"):
            store.put(1, np.zeros((3, 4), dtype=np.uint8), "raw_u8")


class StubBackend:
    output_schema = "test.v1"

    def __init__(self, descriptor):
        self.input_descriptor = descriptor

    def predict(self, batch):
        assert not torch.is_grad_enabled()
        return None


class StubDecoder:
    input_schema = "test.v1"

    def decode(self, output, batch):
        return [
            EndpointResult.from_query(
                sample.query,
                endpoints_xy_px=((8.0, 6.0), (4.0, 3.0)),
                scores={"endpoint_scores": [0.8, 0.3]},
                transform=sample.transform,
            )
            for sample in batch.samples
        ]


def predictor(config=None):
    prep = CropStackPreprocessor(
        config or PreprocessingConfig(temporal_offsets=(-2, -1, 0, 1, 2))
    )
    return EndpointPredictor(prep, StubBackend(prep.descriptor), StubDecoder())


def test_predictor_preserves_order_accounts_for_missing_and_invalid_queries(store):
    good = query()
    missing = replace(good, query_id="missing", context_frame_ids=(0, 1, 2, 3, 999))
    invalid = replace(good, query_id="invalid", bbox_xyxy_px=(1, 1, 1, 2))
    results = predictor().predict(
        store, [missing, good, invalid], pixel_meter_size=0.001
    )
    assert [r.query_id for r in results] == ["missing", "q", "invalid"]
    assert [r.status for r in results] == ["missing_frame", "ok", "invalid_query"]
    assert results[1].endpoints_xy_px == ((4.0, 3.0), (8.0, 6.0))
    assert results[1].scores["endpoint_scores"] == [0.3, 0.8]
    assert results[1].length_pixels == 5
    assert results[1].length_cm == 0.5
    assert results[0].length_pixels is None
    assert results[1].to_dict()["query_id"] == "q"


@pytest.mark.parametrize(
    "field,value",
    [
        ("channels", 3),
        ("target_mask", True),
        ("representations", ("raw_u8",)),
        ("temporal_offsets", (-1, 0, 1)),
    ],
)
def test_predictor_rejects_incompatible_components_at_initialization(field, value):
    prep = CropStackPreprocessor()
    backend = StubBackend(replace(prep.descriptor, **{field: value}))
    with pytest.raises(ValueError, match=field):
        EndpointPredictor(prep, backend, StubDecoder())


def test_predictor_rejects_wrong_decoder_and_missing_representation():
    prep = CropStackPreprocessor()
    backend = StubBackend(prep.descriptor)
    decoder = StubDecoder()
    decoder.input_schema = "different.v1"
    with pytest.raises(ValueError, match="decoder consumes"):
        EndpointPredictor(prep, backend, decoder)
    with pytest.raises(ValueError, match="unavailable representations"):
        EndpointPredictor(
            prep, backend, StubDecoder(), available_representations=("raw_u8",)
        )


def test_backend_errors_propagate_and_do_not_poison_predictor(store):
    instance = predictor()
    original = instance.backend.predict

    def fail(batch):
        raise RuntimeError("systemic failure")

    instance.backend.predict = fail
    with pytest.raises(RuntimeError, match="systemic failure"):
        instance.predict(store, [query()])
    instance.backend.predict = original
    assert instance.predict(store, [query()])[0].status == "ok"


def test_outside_crop_points_flagged_and_nonfinite_outputs_become_misses(store):
    instance = predictor()

    class OutsideDecoder(StubDecoder):
        def decode(self, output, batch):
            return [
                EndpointResult.from_query(
                    sample.query, endpoints_xy_px=((-3, 0), (40, 30))
                )
                for sample in batch.samples
            ]

    instance.decoder = OutsideDecoder()
    result = instance.predict(store, [query()])[0]
    assert set(result.flags) == {"endpoints_outside_crop", "endpoints_outside_frame"}

    class BadDecoder(StubDecoder):
        def decode(self, output, batch):
            return [
                EndpointResult.from_query(
                    sample.query, endpoints_xy_px=((float("nan"), 0), (1, 2))
                )
                for sample in batch.samples
            ]

    instance.decoder = BadDecoder()
    result = instance.predict(store, [query()])[0]
    assert result.status == "no_prediction"
    assert result.endpoints_xy_px is None
    assert result.length_pixels is None


def test_empty_queries_and_cross_clip_isolation(store):
    assert predictor().predict(store, []) == []
    result = predictor().predict(store, [query(clip_id="another")])[0]
    assert result.status == "invalid_query"
    assert result.reason["code"] == "clip_mismatch"


def test_stacking_and_spatial_policies_are_independently_replaceable(store):
    class ReverseStacker(ChannelTemporalStacker):
        component_id = "reverse_temporal_channels.test.v1"

        def describe(self, source):
            return replace(
                source, temporal_offsets=tuple(reversed(source.temporal_offsets))
            )

        def stack(self, crops):
            return super().stack(list(reversed(crops)))

    class OriginalPixels:
        component_id = "original_pixels.test.v1"

        def describe(self, source):
            return replace(source, spatial_mode="native", fixed_hw=None, stride=1)

        def apply(self, image, origin_xy, frame_hw):
            h, w = image.shape[-2:]
            return SpatialPreparedImage(
                image,
                torch.ones((1, h, w), dtype=torch.bool),
                CoordinateTransform(
                    origin_xy, (1.0, 1.0), (0, 0), (h, w), (h, w), frame_hw
                ),
            )

    config = PreprocessingConfig(
        temporal_offsets=(-2, -1, 0, 1, 2), image_size=32, stride=1
    )
    prep = CropStackPreprocessor(
        config, frame_stacker=ReverseStacker(), spatial_transform=OriginalPixels()
    )
    q = query(crop_xyxy_px=(2, 1, 12, 10))
    batch = next(prep.prepare(store, [q]))
    assert batch.images.shape == (1, 5, 9, 10)
    assert batch.images[0, 0, 0, 0] == 200 / 255
    assert batch.images[0, -1, 0, 0] == 0
    assert batch.descriptor.temporal_offsets == (2, 1, 0, -1, -2)
    assert batch.samples[0].transform.canvas_hw == (9, 10)
    # The same predictor/backend/decoder interfaces accept this declared profile.
    assert (
        EndpointPredictor(prep, StubBackend(prep.descriptor), StubDecoder())
        .predict(store, [q])[0]
        .status
        == "ok"
    )
    with pytest.raises(ValueError, match="temporal_offsets"):
        EndpointPredictor(prep, StubBackend(config.descriptor), StubDecoder())
    assert (
        prep.to_dict()["frame_stacker"]["component_id"] == ReverseStacker.component_id
    )


def processor_fixture(tmp_path):
    from fisheye.endpoints.config import EndpointConfig
    from fisheye.endpoints.processor import EndpointProcessor

    prep = CropStackPreprocessor(
        PreprocessingConfig(
            temporal_offsets=(-2, -1, 0, 1, 2),
            spatial_mode="native",
            native_canvas="batch_max",
            stride=1,
            pad_value=0,
            batch_size=2,
        )
    )
    backend = StubBackend(prep.descriptor)
    backend.seen = []
    original_predict = backend.predict

    def capture(batch):
        backend.seen.extend(
            (
                sample.query.clip_id,
                sample.query.frame_id,
                sample.query.context_frame_ids,
            )
            for sample in batch.samples
        )
        return original_predict(batch)

    backend.predict = capture

    class BoxCornerDecoder(StubDecoder):
        def decode(self, output, batch):
            return [
                EndpointResult.from_query(
                    sample.query,
                    endpoints_xy_px=tuple(
                        tuple(float(x) for x in point)
                        for point in sample.transform.to_full(
                            sample.target_box.reshape(2, 2).flip(0)
                        )
                    ),
                    scores={"keypoint_confidence": [0.8, 0.3]},
                    transform=sample.transform,
                )
                for sample in batch.samples
            ]

    instance = EndpointPredictor(prep, backend, BoxCornerDecoder())
    config = EndpointConfig(
        enabled=True,
        model={"bundle": "injected"},
        frame_store={
            "ram_limit_mb": 0,
            "disk_limit_gb": 0.01,
            "directory": str(tmp_path / "scratch"),
        },
    )
    processor = EndpointProcessor(
        config,
        predictor=instance,
        spec={"image_preparation": {"use_blur": False, "only_positive": False}},
    )
    return processor, backend


def capture_processor_clip(session):
    description = {
        "id": "test_unblurred_signed.v1",
        "representation": "bgs_signed_u8",
        "use_blur": False,
        "only_positive": False,
    }
    session.capture(
        {
            "frame_ids": (5, 25),
            "images": [np.full((13, 21), value, np.uint8) for value in (10, 20)],
            "preprocessing": description,
        },
        batch_index=0,
        batch_size=2,
    )
    session.capture(
        {
            "frame_ids": (90,),
            "images": [np.full((13, 21), 30, np.uint8)],
            "preprocessing": description,
        },
        batch_index=1,
        batch_size=1,
    )
    boxes = np.array(
        [[2 / 21, 2 / 13, 6 / 21, 6 / 13, 0.9], [12 / 21, 5 / 13, 17 / 21, 9 / 13, 0.8]]
    )
    low = {key: boxes.copy() for key in session.frame_ids}
    high = {key: boxes.copy() for key in session.frame_ids}
    fish = lambda track, index: SimpleNamespace(
        id=track,
        det_index=1 if index is not None else None,
        bbox_index=index,
        bbox=(0, 0, 0, 0),
    )
    # Deliberately reverse the tracker frame order. The middle observation for
    # fish 7 is a tracker extrapolation and must not enter temporal selection.
    tracks = SimpleNamespace(
        frames=[
            SimpleNamespace(frame_num=90, fish=[fish(7, 0), fish(9, 1)]),
            SimpleNamespace(frame_num=25, fish=[fish(7, None), fish(9, 1)]),
            SimpleNamespace(frame_num=5, fish=[fish(7, 0), fish(9, 1)]),
        ]
    )
    metadata = SimpleNamespace(xdim=21, ydim=13, pixel_meter_size=0.01)
    return tracks, low, high, metadata


def test_processor_capture_track_queries_prediction_and_atomic_sidecar(tmp_path):
    processor, backend = processor_fixture(tmp_path)
    output = tmp_path / "output"
    with processor.session("clip") as session:
        scratch = session.store.directory
        tracks, low, high, metadata = capture_processor_clip(session)
        results = processor.process(
            session,
            tracks,
            low,
            high,
            metadata,
            output,
            "fish",
            tracker_config={"reverse": True},
        )
        assert len(results) == 5
        assert {(r.track_id, r.frame_id) for r in results} == {
            (7, 5),
            (7, 90),
            (9, 5),
            (9, 25),
            (9, 90),
        }
        first = results[0]
        assert first.context_frame_ids == (5, 5, 5, 90, 90)
        assert first.context_repeated == (True, True, False, False, True)
        np.testing.assert_allclose(first.endpoints_xy_px, ((2, 2), (6, 6)))
        assert first.length_pixels == pytest.approx(32**0.5)
        assert first.length_cm == pytest.approx(32**0.5)
        assert first.scores["keypoint_confidence"] == [0.3, 0.8]
        assert session.store.ram_bytes == 0
        assert session.store.disk_bytes > 0
        document = json.loads((output / "fish.endpoints.json").read_text())
        assert document["counts"]["queries"] == 5
        assert document["counts"]["skipped_observations"] == 1
        assert document["counts"]["frames"] == 3
        assert document["skipped_observations"][0]["frame_id"] == 25
        assert document["predictions"][0]["transform"]["frame_hw"] == [13, 21]
        assert document["tracker_config"]["reverse"] is True
        assert len(backend.seen) == 5
        assert all(list(context) == sorted(context) for _, _, context in backend.seen)
        assert list(output.iterdir()) == [output / "fish.endpoints.json"]
    assert not scratch.exists()


def test_processor_failure_cleans_store_and_next_clip_has_fresh_state(tmp_path):
    processor, backend = processor_fixture(tmp_path)
    output = tmp_path / "output"
    original = backend.predict

    def fail(batch):
        raise RuntimeError("GPU inference failed")

    backend.predict = fail
    with pytest.raises(RuntimeError, match="GPU inference failed"):
        with processor.session("bad") as session:
            bad_scratch = session.store.directory
            tracks, low, high, metadata = capture_processor_clip(session)
            processor.process(session, tracks, low, high, metadata, output, "bad")
    assert not bad_scratch.exists()
    assert not (output / "bad.endpoints.json").exists()
    backend.predict = original
    with processor.session("good") as session:
        assert session.frame_ids == {}
        assert session.store.frame_ids == ()
        tracks, low, high, metadata = capture_processor_clip(session)
        results = processor.process(
            session, tracks, low, high, metadata, output, "good"
        )
    assert all(r.clip_id == "good" for r in results)
    assert all(clip == "good" for clip, _, _ in backend.seen)
    assert not list((tmp_path / "scratch").iterdir())

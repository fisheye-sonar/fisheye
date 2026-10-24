"""Clip orchestration and atomic exports; model code stays independent of FishEye tracks."""

from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

from .config import EndpointConfig
from .aggregation import aggregate_track_lengths
from .cone import SonarConeGeometry
from .frame_store import BoundedFrameStore, RepresentationDescriptor
from .queries import build_queries, detection_records
from .temporal import TrackTemporalSelector


class EndpointSession:
    def __init__(self, store):
        self.store = store
        self.frame_ids = {}
        self.preprocessing = None
        self.capture_seconds = 0.0

    def capture(self, payload, batch_index, batch_size):
        started = perf_counter()
        ids, images = payload["frame_ids"], payload["images"]
        if len(ids) != batch_size or len(images) != batch_size:
            raise ValueError("Endpoint capture must identify every batch image")
        descriptor = payload["preprocessing"]
        if self.preprocessing is not None and descriptor != self.preprocessing:
            raise ValueError("Endpoint preprocessing changed within a clip")
        self.preprocessing = descriptor
        for local_index, (frame_id, image) in enumerate(zip(ids, images)):
            key = (batch_index, local_index)
            if key in self.frame_ids:
                raise ValueError("Duplicate endpoint detection frame key")
            self.store.put(
                int(frame_id),
                image,
                representation=descriptor["representation"],
            )
            self.frame_ids[key] = int(frame_id)
        self.capture_seconds += perf_counter() - started


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".tmp",
            prefix=path.name + ".",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                document, handle, allow_nan=False, indent=2, default=_json_default
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_default(value):
    from enum import Enum

    if isinstance(value, (Path, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    raise TypeError(f"Cannot serialize endpoint metadata value {type(value).__name__}")


class EndpointProcessor:
    def __init__(self, config, predictor=None, spec=None):
        self.config = EndpointConfig.from_value(config)
        if predictor is None:
            from .bundle import load_bundle
            from .preprocessing import CropStackPreprocessor, PreprocessingConfig
            from .predictor import EndpointPredictor

            loaded = load_bundle(self.config.model["bundle"], device=self.config.device)
            self.spec = loaded.spec
            profile = PreprocessingConfig(**loaded.preprocessing)
            profile = replace(profile, batch_size=self.config.crop_batch_size)
            preprocessor = CropStackPreprocessor(profile)
            predictor = EndpointPredictor(preprocessor, loaded.backend, loaded.decoder)
        else:
            self.spec = dict(spec or {})
        self.predictor = predictor
        self.selector = TrackTemporalSelector(
            offsets=predictor.preprocessor.config.temporal_offsets,
        )
        self.representation = predictor.preprocessor.config.representation
        if self.representation != "bgs_signed_u8":
            raise ValueError(
                "FishEye endpoint inference requires bgs_signed_u8 from detector channel 1"
            )
        preparation = self.spec.get("image_preparation", {})
        if (
            preparation.get("use_blur", False)
            or preparation.get("only_positive", False) is not False
        ):
            raise ValueError(
                "Endpoint image preparation disagrees with the selected unblurred BGS representation"
            )
        self.last_document = None
        self.last_length_outputs = {}

    @contextmanager
    def session(self, clip_id):
        settings = self.config.frame_store
        with BoundedFrameStore(
            clip_id,
            ram_limit_bytes=int(settings.get("ram_limit_mb", 512) * 1024**2),
            disk_limit_bytes=int(settings.get("disk_limit_gb", 20) * 1024**3),
            directory=settings.get("directory"),
            representations=(RepresentationDescriptor(self.representation),),
        ) as store:
            yield EndpointSession(store)

    def process(
        self,
        session,
        tracker_output,
        low_preds,
        high_preds,
        metadata,
        output_dir,
        filename_stem,
        tracker_config=None,
        detector_provenance=None,
    ):
        self.last_document = None
        self.last_length_outputs = {}
        started = perf_counter()
        if session.store.frame_hw is not None:
            height, width = session.store.frame_hw
            if (height, width) != (metadata.ydim, metadata.xdim):
                raise ValueError("Endpoint frames and detector image dimensions differ")
        else:
            height, width = metadata.ydim, metadata.xdim
        records = detection_records(
            low_preds, high_preds, session.frame_ids, width, height
        )
        queries, skipped = build_queries(
            tracker_output, records, session.store.clip_id, self.selector
        )
        query_seconds = perf_counter() - started
        predicted_at = perf_counter()
        results = self.predictor.predict(
            session.store, queries, pixel_meter_size=metadata.pixel_meter_size
        )
        counts = {}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
        aggregation = None
        if self.config.track_length.enabled:
            cone = SonarConeGeometry.from_metadata(metadata)
            aggregation = aggregate_track_lengths(
                results,
                (height, width),
                self.config.track_length,
                cone=cone,
                track_ids={
                    fish.id for frame in tracker_output.frames for fish in frame.fish
                },
            )
        self.last_document = {
            "schema_version": 1,
            "status": "complete",
            "clip_id": session.store.clip_id,
            "inference_spec": self.spec,
            "resolved_preprocessing": asdict(self.predictor.preprocessor.config),
            "components": self.predictor.provenance(),
            "temporal_selector": self.selector.to_dict(),
            "image_preparation": session.preprocessing,
            "tracker_config": tracker_config or {},
            "detector": detector_provenance or {},
            "selection": "retained tracks, matched detections only",
            "pixel_meter_size": metadata.pixel_meter_size,
            "counts": {
                "frames": len(session.frame_ids),
                "detection_records": len(records),
                "queries": len(queries),
                "skipped_observations": len(skipped),
                "repeated_context_positions": sum(
                    sum(q.context_repeated) for q in queries
                ),
                "results": counts,
            },
            "skipped_observations": skipped,
            "frame_store": session.store.metadata(),
            "timings_seconds": {
                "capture": session.capture_seconds,
                "queries": query_seconds,
                "prediction": perf_counter() - predicted_at,
            },
            "predictions": [result.to_dict() for result in results],
        }
        if aggregation is not None:
            self.last_document["sonar_cone"] = cone.to_dict()
            self.last_document["track_length_config"] = asdict(self.config.track_length)
            self.last_document["track_lengths"] = aggregation.tracks
            for prediction, decision in zip(
                self.last_document["predictions"], aggregation.prediction_filters
            ):
                prediction["length_filter"] = decision
            self.last_document["counts"]["track_lengths"] = {
                "accepted": len(aggregation.length_outputs),
                "discarded": len(aggregation.tracks) - len(aggregation.length_outputs),
            }
        if self.config.export_predictions:
            atomic_json(
                Path(output_dir) / f"{filename_stem}.endpoints.json", self.last_document
            )
        if aggregation is not None:
            self.last_length_outputs = aggregation.length_outputs
        return results

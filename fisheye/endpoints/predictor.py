"""Persistent inference orchestration with independently injected components."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import math
import threading
from typing import Iterable

import numpy as np
import torch

from .types import (
    EndpointQuery,
    EndpointResult,
    FrameStore,
    InputPreprocessor,
    ModelBackend,
    OutputDecoder,
    QueryValidationError,
)


class EndpointPredictor:
    """Reuse model state across clips; each call owns its frame/query metadata.

    Invalid queries are accounted for individually. Unexpected preprocessing,
    backend, decoder, and resource errors propagate so the clip cannot publish
    empty successful output. Concurrent calls are rejected rather than mixed.
    """

    def __init__(
        self,
        preprocessor: InputPreprocessor,
        backend: ModelBackend,
        decoder: OutputDecoder,
        available_representations=None,
    ):
        preprocessor.descriptor.validate_compatible(backend.input_descriptor)
        if backend.output_schema != decoder.input_schema:
            raise ValueError(
                f"Endpoint decoder consumes {decoder.input_schema!r}, "
                f"but backend produces {backend.output_schema!r}"
            )
        if available_representations is not None:
            missing = set(preprocessor.descriptor.representations) - set(
                available_representations
            )
            if missing:
                raise ValueError(
                    f"Endpoint preprocessing requires unavailable representations: {sorted(missing)}"
                )
        self.preprocessor, self.backend, self.decoder = preprocessor, backend, decoder
        self._inference_lock = threading.Lock()

    @classmethod
    def from_bundle(cls, bundle_path, device="cpu", batch_size=None):
        from .bundle import load_bundle
        from .preprocessing import CropStackPreprocessor, PreprocessingConfig

        loaded = load_bundle(bundle_path, device=device)
        config = dict(loaded.preprocessing)
        if batch_size is not None:
            config["batch_size"] = batch_size
        return cls(
            CropStackPreprocessor(PreprocessingConfig(**config)),
            loaded.backend,
            loaded.decoder,
        )

    @staticmethod
    def _finalize(result: EndpointResult, pixel_meter_size):
        if result.status not in {
            "ok",
            "no_prediction",
            "invalid_query",
            "missing_frame",
            "inference_error",
        }:
            raise ValueError(
                f"Decoder returned unsupported endpoint status {result.status!r}"
            )
        if result.status != "ok":
            return replace(
                result, endpoints_xy_px=None, length_pixels=None, length_cm=None
            )
        points = np.asarray(result.endpoints_xy_px, dtype=float)
        if points.shape != (2, 2) or not np.isfinite(points).all():
            return replace(
                result,
                status="no_prediction",
                endpoints_xy_px=None,
                length_pixels=None,
                length_cm=None,
                reason={
                    "code": "nonfinite_endpoints",
                    "message": "Decoder did not return two finite endpoints",
                },
            )
        order = np.argsort(points[:, 0], kind="stable")
        points = points[order]
        scores = dict(result.scores)
        for name, value in scores.items():
            # Score semantics stay with the decoder. Only explicitly point-wise
            # scores are reordered; an unrelated pair of scalar diagnostics is not.
            if name in {
                "endpoint_scores",
                "keypoint_confidence",
                "endpoint_confidence",
                "heatmap_peaks",
                "endpoint_entropy",
                "endpoint_peak",
            }:
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
                    scores[name] = [value[int(i)] for i in order]
        flags = list(result.flags)
        transform = result.transform
        if transform is not None:
            x, y = transform.origin_xy
            h, w = transform.crop_hw
            if ((points < (x, y)) | (points >= (x + w, y + h))).any():
                flags.append("endpoints_outside_crop")
            if transform.frame_hw is not None:
                fh, fw = transform.frame_hw
                if ((points < 0) | (points >= (fw, fh))).any():
                    flags.append("endpoints_outside_frame")
        length = float(np.linalg.norm(points[1] - points[0]))
        return replace(
            result,
            endpoints_xy_px=tuple(tuple(float(x) for x in point) for point in points),
            length_pixels=length,
            length_cm=(
                length * pixel_meter_size * 100
                if pixel_meter_size is not None
                else None
            ),
            scores=scores,
            flags=tuple(dict.fromkeys(flags)),
        )

    def predict(
        self,
        frame_store: FrameStore,
        queries: Iterable[EndpointQuery],
        pixel_meter_size: float | None = None,
    ) -> list[EndpointResult]:
        if pixel_meter_size is not None and (
            not math.isfinite(pixel_meter_size) or pixel_meter_size <= 0
        ):
            raise ValueError(
                "pixel_meter_size must be positive and finite when supplied"
            )
        if pixel_meter_size is not None:
            pixel_meter_size = float(pixel_meter_size)
        if not self._inference_lock.acquire(blocking=False):
            raise RuntimeError(
                "Concurrent calls to one EndpointPredictor are unsupported"
            )
        try:
            return self._predict(frame_store, list(queries), pixel_meter_size)
        finally:
            self._inference_lock.release()

    def _predict(self, frame_store, queries, pixel_meter_size):
        ids = Counter(q.query_id for q in queries)
        results, valid = {}, []
        for query in queries:
            try:
                if ids[query.query_id] != 1:
                    raise QueryValidationError(
                        "duplicate_query_id",
                        "query_id must be unique within one prediction call",
                    )
                if query.clip_id != frame_store.clip_id:
                    raise QueryValidationError(
                        "clip_mismatch",
                        "Query and frame store belong to different clips",
                    )
                if hasattr(self.preprocessor, "validate_query"):
                    self.preprocessor.validate_query(frame_store, query)
            except QueryValidationError as exc:
                results[query.query_id] = EndpointResult.from_query(
                    query,
                    status=exc.status,
                    reason={"code": exc.code, "message": str(exc)},
                )
            else:
                valid.append(query)
        expected = {q.query_id: q for q in valid}
        for batch in self.preprocessor.prepare(frame_store, valid):
            batch.descriptor.validate_compatible(self.backend.input_descriptor)
            if (
                batch.images.ndim != 4
                or batch.images.shape[0] != len(batch.samples)
                or batch.images.shape[1] != batch.descriptor.channels
                or batch.images.dtype != torch.float32
            ):
                raise ValueError(
                    "Endpoint preprocessor produced invalid BCHW float32 images"
                )
            if (
                batch.support.shape
                != (batch.images.shape[0], 1, *batch.images.shape[-2:])
                or batch.support.dtype != torch.bool
            ):
                raise ValueError(
                    "Endpoint preprocessor produced invalid B1HW boolean support"
                )
            if (
                batch.descriptor.fixed_hw is not None
                and tuple(batch.images.shape[-2:]) != batch.descriptor.fixed_hw
            ):
                raise ValueError(
                    "Endpoint preprocessor canvas does not match the model's fixed size"
                )
            if any(size % batch.descriptor.stride for size in batch.images.shape[-2:]):
                raise ValueError(
                    "Endpoint preprocessor canvas does not match the model stride"
                )
            if (
                not torch.isfinite(batch.images).all()
                or batch.images.min() < batch.descriptor.value_range[0]
                or batch.images.max() > batch.descriptor.value_range[1]
            ):
                raise ValueError(
                    "Endpoint preprocessor produced pixels outside its declared value range"
                )
            with torch.inference_mode():
                output = self.backend.predict(batch)
                decoded = self.decoder.decode(output, batch)
            if len(decoded) != len(batch.samples):
                raise ValueError(
                    "Endpoint decoder must return exactly one result per prepared sample"
                )
            for sample, result in zip(batch.samples, decoded):
                if (
                    result.query_id != sample.query.query_id
                    or result.query_id not in expected
                    or result.query_id in results
                ):
                    raise ValueError(
                        "Endpoint decoder changed, duplicated, or reordered query identities"
                    )
                for name in (
                    "clip_id",
                    "frame_id",
                    "track_id",
                    "detection_id",
                    "bbox_xyxy_px",
                    "context_frame_ids",
                    "context_repeated",
                ):
                    if getattr(result, name) != getattr(sample.query, name):
                        raise ValueError(
                            f"Endpoint decoder changed query identity field {name}"
                        )
                if result.transform is None:
                    result = replace(result, transform=sample.transform)
                results[result.query_id] = self._finalize(result, pixel_meter_size)
        missing = set(expected) - set(results)
        if missing:
            raise ValueError(
                f"Endpoint preprocessing omitted valid queries: {sorted(missing)}"
            )
        return [results[query.query_id] for query in queries]

    def provenance(self):
        return {
            "preprocessor": (
                self.preprocessor.to_dict()
                if hasattr(self.preprocessor, "to_dict")
                else {
                    "component_id": getattr(
                        self.preprocessor,
                        "component_id",
                        type(self.preprocessor).__name__,
                    ),
                    "descriptor": asdict(self.preprocessor.descriptor),
                }
            ),
            "backend": {
                "component_id": getattr(
                    self.backend, "component_id", type(self.backend).__name__
                ),
                "output_schema": self.backend.output_schema,
            },
            "decoder": {
                "component_id": getattr(
                    self.decoder, "component_id", type(self.decoder).__name__
                ),
                "input_schema": self.decoder.input_schema,
            },
        }

"""Modular endpoint inference; model dependencies load only when requested."""

from importlib import import_module

_EXPORTS = {
    "EndpointPredictor": "predictor",
    "EndpointQuery": "types",
    "EndpointResult": "types",
    "InputDescriptor": "types",
    "CoordinateTransform": "types",
    "PreparedBatch": "types",
    "PreparedSample": "types",
    "CropStackPreprocessor": "preprocessing",
    "PreprocessingConfig": "preprocessing",
    "TrackTemporalSelector": "temporal",
    "ResolvedContext": "temporal",
    "BoundedFrameStore": "frame_store",
    "ChannelTemporalStacker": "preprocessing",
    "ReferenceSpatialTransform": "preprocessing",
    "TemporalStacker": "types",
    "SpatialTransform": "types",
    "SpatialPreparedImage": "types",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value

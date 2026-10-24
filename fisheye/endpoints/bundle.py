"""Portable endpoint bundles: explicit profiles, clean weights, and verified hashes.

Export trusted research checkpoints once, then deploy only the resulting directory.
Runtime loading uses ``weights_only=True`` and never imports training code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from .backends import (
    BACKEND_SCHEMAS,
    ULTRALYTICS_VERSION,
    TorchModelBackend,
    descriptor_from_preprocessing,
)
from .decoders import create_decoder

SCHEMA_VERSION = 1
SOURCE_REVISION = "e29d4614ce523f483027bc4a61b8cb5993fade11"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _primitive(value):
    """Drop class paths from configuration while preserving JSON scalar types."""
    if isinstance(value, dict):
        return {str(k): _primitive(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_primitive(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(
        f"Checkpoint configuration contains unsupported value {type(value).__name__}"
    )


def _resolve_checkpoint(source):
    import torch

    # Export is an explicit trust boundary for local research checkpoints, which
    # include optimizer state and NumPy RNG objects. These are never deployed.
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a saved config and state dictionary")
    config = _primitive(payload["config"])
    radius = config["dataset"].get("num_surrounding_frames", 2)
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ValueError("Checkpoint temporal radius must be a nonnegative integer")
    mode = config.get("experiment", {}).get("mode")
    if mode is not None and mode not in (
        "all_boxes",
        "center_pose",
        "target_pose",
        "target_heatmap",
    ):
        raise ValueError(f"Unsupported checkpoint experiment mode {mode!r}")
    if mode is not None and payload.get("experiment_format") != 1:
        raise ValueError(
            "Expected experiment_format=1 for endpoint experiment checkpoint"
        )
    if "model_state" in payload:
        backend = "yolo_target_heatmap" if mode == "target_heatmap" else "yolo_pose"
        architecture = payload.get("model_yaml")
        if not isinstance(architecture, dict):
            raise ValueError(
                "YOLO checkpoint must embed model_yaml; architecture filenames are insufficient"
            )
        state = payload["model_state"]
    elif "model_state_dict" in payload:
        model_type = config["model"].get("type")
        if model_type not in ("unet", "heatmap_cnn"):
            raise ValueError(f"Unsupported native heatmap model type {model_type!r}")
        backend = "unet_heatmap" if model_type == "unet" else "cnn_heatmap"
        architecture = None
        state = payload["model_state_dict"]
    else:
        raise ValueError("Checkpoint lacks model_state or model_state_dict")
    channels = 2 * radius + 1 + int(mode in ("target_pose", "target_heatmap"))
    if config["dataset"].get("use_rgb_input", False):
        raise ValueError(
            "Initial portable profiles require one BGS channel per temporal frame"
        )
    stem = (
        "features.0.conv.weight"
        if backend == "yolo_target_heatmap"
        else (
            "model.0.conv.weight"
            if backend == "yolo_pose"
            else "inc.block.0.weight" if backend == "unet_heatmap" else "down.0.weight"
        )
    )
    if stem not in state or state[stem].shape[1] != channels:
        raise ValueError(
            f"Checkpoint stem {stem!r} disagrees with saved temporal/mask channel count {channels}"
        )
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise ValueError("Model state must contain only named tensors")
    clean = {
        "bundle_weights_format": 1,
        "backend": backend,
        "input_channels": channels,
        "config": config,
        "architecture": _primitive(architecture),
        "model_state": {k: v.detach().cpu() for k, v in state.items()},
    }
    return clean


def _validate_profile(profile, payload):
    if not isinstance(profile, dict):
        raise ValueError("Profile must be a mapping")
    preprocessing = dict(profile.get("preprocessing", {}))
    required = (
        "representation",
        "temporal_offsets",
        "crop_expansion",
        "spatial_mode",
        "image_size",
        "stride",
        "target_mask",
        "pad_value",
        "native_canvas",
    )
    missing = [name for name in required if name not in preprocessing]
    if missing:
        raise ValueError(
            f"Profile must explicitly resolve preprocessing fields: {missing}"
        )
    # Use the same constructor as the runtime, including radius/offset, canvas,
    # image-size, and type validation. Batch size is a runtime resource setting.
    from .preprocessing import PreprocessingConfig

    descriptor = descriptor_from_preprocessing(preprocessing)
    PreprocessingConfig(**preprocessing)
    config = payload["config"]
    radius = config["dataset"].get("num_surrounding_frames", 2)
    offsets = tuple(range(-radius, radius + 1))
    if descriptor.temporal_offsets != offsets:
        raise ValueError(f"Checkpoint expects chronological temporal offsets {offsets}")
    if not isinstance(preprocessing["target_mask"], bool):
        raise ValueError("Profile target_mask must be a boolean")
    mode = config.get("experiment", {}).get("mode")
    if descriptor.target_mask != (mode in ("target_pose", "target_heatmap")):
        raise ValueError(
            "Profile target_mask disagrees with checkpoint experiment mode"
        )
    if descriptor.channels != payload["input_channels"]:
        raise ValueError(
            "Profile temporal stack/mask count differs from checkpoint channels"
        )
    if payload["backend"].startswith("yolo_"):
        spatial = config["training"].get("input_resolution", "letterbox")
        if preprocessing["spatial_mode"] != spatial:
            raise ValueError(f"Checkpoint uses {spatial!r} spatial preprocessing")
        expected_size = config["training"].get("imgsz")
        if spatial == "letterbox" and descriptor.fixed_hw != (
            expected_size,
            expected_size,
        ):
            raise ValueError(f"Checkpoint requires letterbox size {expected_size}")
        if preprocessing["stride"] != 32 or preprocessing["pad_value"] != 114 / 255:
            raise ValueError(
                "Reference YOLO profile requires stride=32 and pad_value=114/255"
            )
    elif preprocessing["spatial_mode"] != "native":
        raise ValueError(
            "Native heatmap checkpoints require native spatial preprocessing"
        )
    image_preparation = profile.get("image_preparation")
    required_image = (
        "id",
        "coordinate_space",
        "use_blur",
        "only_positive",
        "quantization",
        "background_statistics",
        "source_value_range",
    )
    if not isinstance(image_preparation, dict) or any(
        k not in image_preparation for k in required_image
    ):
        raise ValueError(
            f"Profile image_preparation must explicitly provide {required_image}"
        )
    representation = preprocessing["representation"]
    if representation not in ("bgs_positive_u8", "bgs_signed_u8"):
        raise ValueError(
            f"Unsupported endpoint image representation {representation!r}"
        )
    positive = representation == "bgs_positive_u8"
    if (
        image_preparation["coordinate_space"] != "warped_original"
        or image_preparation["use_blur"] is not False
        or image_preparation["only_positive"] is not positive
        or image_preparation["quantization"] != "uint8"
        or image_preparation["source_value_range"] != [0, 255]
    ):
        raise ValueError(
            "Endpoint source profile requires original warped, unblurred uint8 BGS "
            "with matching signed/positive encoding"
        )
    if not positive and (
        image_preparation.get("source") != "detector_center_channel"
        or image_preparation.get("source_channel") != 1
    ):
        raise ValueError(
            "Signed BGS profiles must select detector_center_channel source_channel=1"
        )
    return preprocessing, _primitive(image_preparation)


def _decoder_config(payload, profile):
    config = payload["config"]
    if payload["backend"] == "yolo_pose":
        decoder = {
            "name": "yolo_pose",
            "candidate_confidence": config["evaluation"]["candidate_confidence"],
            "iou_threshold": 0.7,
            "max_det": 300,
        }
    elif payload["backend"] == "yolo_target_heatmap":
        decoder = {"name": "target_heatmap"}
    else:
        if "decoder" not in profile:
            raise ValueError(
                "Native heatmap profile requires an explicit decoder configuration; "
                "historical GT-dependent post-forward cropping cannot be inferred"
            )
        decoder = dict(profile["decoder"])
        if decoder.get("name") != "native_heatmap":
            raise ValueError("Native heatmap backend requires native_heatmap decoder")
    # Overrides are explicit experiment profiles and must remain schema compatible.
    if "decoder" in profile:
        decoder = dict(profile["decoder"])
    instance = create_decoder(decoder)
    if instance.input_schema != BACKEND_SCHEMAS[payload["backend"]]:
        raise ValueError("Decoder output family is incompatible with model backend")
    return decoder


def export_bundle(checkpoint, destination, profile, *, source_revision=SOURCE_REVISION):
    """Export a trusted local checkpoint and explicitly resolved source/crop profile."""
    import torch

    checkpoint, destination = Path(checkpoint), Path(destination)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing bundle {destination}")
    source_hash = sha256_file(checkpoint)
    payload = _resolve_checkpoint(checkpoint)
    preprocessing, image_preparation = _validate_profile(profile, payload)
    decoder = _decoder_config(payload, profile)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        weights = staging / "weights.pt"
        torch.save(payload, weights)
        backend = payload["backend"]
        spec = {
            "schema_version": SCHEMA_VERSION,
            "model": {
                "backend": backend,
                "input_channels": payload["input_channels"],
                "output_schema": BACKEND_SCHEMAS[backend],
                "checkpoint": weights.name,
                "checkpoint_sha256": sha256_file(weights),
            },
            "preprocessing": _primitive(preprocessing),
            "image_preparation": image_preparation,
            "decoder": _primitive(decoder),
            "temporal_selection": {
                "basis": "matched_track_positions",
                "boundary": "repeat_edge",
                "offsets": preprocessing["temporal_offsets"],
            },
            "endpoint_order": "increasing_x_stable",
            "source": {
                "checkpoint_name": checkpoint.name,
                "checkpoint_sha256": source_hash,
                "repository_revision": source_revision,
                "experiment_mode": payload["config"].get("experiment", {}).get("mode"),
            },
            "runtime": {
                "torch": ">=2.6",
                **(
                    {"ultralytics": ULTRALYTICS_VERSION}
                    if backend.startswith("yolo_")
                    else {}
                ),
            },
            "components": {
                "preprocessor": "crop_stack.v1",
                "backend": backend + ".v1",
                "decoder": decoder["name"] + ".v1",
            },
            "profile_notes": _primitive(profile.get("notes", {})),
        }
        (staging / "inference_spec.json").write_text(
            json.dumps(spec, indent=2, allow_nan=False) + "\n"
        )
        # Validate actual model construction and exact state compatibility before
        # making a reviewable bundle visible. No pretrained assets are requested.
        load_bundle(staging, device="cpu")
        if sha256_file(checkpoint) != source_hash:
            raise ValueError("Source checkpoint changed during export")
        staging.replace(destination)
        return spec
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_bundle(path):
    """Validate JSON schema/hash and checkpoint/profile compatibility without a forward pass."""
    import torch

    path = Path(path)
    spec = json.loads((path / "inference_spec.json").read_text())
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f'Unsupported endpoint bundle schema {spec.get("schema_version")!r}'
        )
    for name in (
        "model",
        "preprocessing",
        "image_preparation",
        "decoder",
        "source",
        "runtime",
        "components",
    ):
        if not isinstance(spec.get(name), dict):
            raise ValueError(f"Bundle spec must contain mapping {name!r}")
    model = spec["model"]
    if model.get("backend") not in BACKEND_SCHEMAS:
        raise ValueError(f'Unknown endpoint backend {model.get("backend")!r}')
    relative = Path(model.get("checkpoint", ""))
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name in ("", ".", "..")
    ):
        raise ValueError("Bundle checkpoint must be a local basename")
    weights = path / relative
    if weights.is_symlink():
        raise ValueError("Portable bundle checkpoint must not be a symlink")
    if sha256_file(weights) != model.get("checkpoint_sha256"):
        raise ValueError("Endpoint checkpoint SHA256 mismatch")
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    if payload.get("bundle_weights_format") != 1:
        raise ValueError("Unsupported portable checkpoint format")
    if payload.get("backend") != model["backend"] or payload.get(
        "input_channels"
    ) != model.get("input_channels"):
        raise ValueError(
            "Bundle spec backend/channels disagree with embedded checkpoint contract"
        )
    if model.get("output_schema") != BACKEND_SCHEMAS[model["backend"]]:
        raise ValueError("Bundle output schema disagrees with model backend")
    _validate_profile(spec, payload)
    selection = spec.get("temporal_selection", {})
    if (
        selection.get("basis") != "matched_track_positions"
        or selection.get("boundary") != "repeat_edge"
        or selection.get("offsets") != spec["preprocessing"]["temporal_offsets"]
    ):
        raise ValueError(
            "Bundle temporal selection must declare matched track positions, "
            "repeat_edge boundaries, and the preprocessing offsets"
        )
    if spec.get("endpoint_order") != "increasing_x_stable":
        raise ValueError("Bundle must declare stable increasing-x endpoint ordering")
    decoder = create_decoder(spec["decoder"])
    if decoder.input_schema != model["output_schema"]:
        raise ValueError("Bundle decoder is incompatible with model output schema")
    if (
        model["backend"].startswith("yolo_")
        and spec["runtime"].get("ultralytics") != ULTRALYTICS_VERSION
    ):
        raise ValueError("Bundle must declare its pinned Ultralytics version")
    expected_components = {
        "preprocessor": "crop_stack.v1",
        "backend": model["backend"] + ".v1",
        "decoder": spec["decoder"]["name"] + ".v1",
    }
    if spec["components"] != expected_components:
        raise ValueError(
            "Bundle component identifiers/versions disagree with its resolved configuration"
        )
    return spec


@dataclass
class LoadedBundle:
    spec: dict
    backend: TorchModelBackend
    decoder: object
    preprocessing: dict


def load_bundle(path, *, device="cpu"):
    import torch

    path = Path(path)
    spec = validate_bundle(path)
    payload = torch.load(
        path / spec["model"]["checkpoint"], map_location="cpu", weights_only=True
    )
    backend = TorchModelBackend(payload, spec["preprocessing"], device=device)
    return LoadedBundle(
        spec, backend, create_decoder(spec["decoder"]), dict(spec["preprocessing"])
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser(
        "export", help="Export a trusted research checkpoint"
    )
    exporter.add_argument("checkpoint", type=Path)
    exporter.add_argument("destination", type=Path)
    exporter.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="JSON with explicit preprocessing and image_preparation semantics",
    )
    exporter.add_argument("--source-revision", default=SOURCE_REVISION)
    validator = commands.add_parser(
        "validate", help="Check schema, hash, and model loading"
    )
    validator.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        spec = export_bundle(
            args.checkpoint,
            args.destination,
            json.loads(args.profile.read_text()),
            source_revision=args.source_revision,
        )
    else:
        spec = load_bundle(args.bundle).spec
    print(
        json.dumps(
            {"schema_version": spec["schema_version"], **spec["model"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()

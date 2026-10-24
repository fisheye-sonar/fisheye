"""Development-only parity check against explicitly supplied research sources.

Run from the FishEye checkout with::

    python -m scripts.check_endpoint_reference_parity \
        --research-root /path/to/length_annotator \
        --output docs/endpoint_model_parity.json

The deployed endpoint package does not import these research sources. This tool
loads trusted local research checkpoints and reviewed images to compare the
ported image tensors, model outputs, matching, and inverse coordinate mapping.
It never downloads assets or ranks/tunes checkpoints using these queries.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SELECTED = {
    "yolo_5f": "runs/yolo_pose/yolo26s_pose_manual_lr1e4_gpu1/best_percentage_length_error.pt",
    "heatmap_5f": "runs/endpoint_experiments/target_heatmap/best_percentage_length_error.pt",
}


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument(
        "--yolo-bundle", type=Path, default=REPOSITORY / "weights/endpoints/yolo_5f"
    )
    parser.add_argument(
        "--heatmap-bundle",
        type=Path,
        default=REPOSITORY / "weights/endpoints/heatmap_5f",
    )
    parser.add_argument(
        "--yolo-checkpoint", type=Path, default=Path(SELECTED["yolo_5f"])
    )
    parser.add_argument(
        "--heatmap-checkpoint", type=Path, default=Path(SELECTED["heatmap_5f"])
    )
    parser.add_argument("--role", choices=("val", "test"), default="val")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("yolo_5f", "heatmap_5f", "unblurred_signed_bgs_1f"),
        default=["unblurred_signed_bgs_1f"],
    )
    parser.add_argument("--signed-checkpoint", type=Path)
    parser.add_argument(
        "--signed-bundle",
        type=Path,
        default=REPOSITORY / "weights/endpoints/unblurred_signed_bgs_1f_seed2",
    )
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.queries < 3:
        parser.error("--queries must be at least 3")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if "unblurred_signed_bgs_1f" in args.models and args.signed_checkpoint is None:
        parser.error(
            "--signed-checkpoint is required for unblurred_signed_bgs_1f parity"
        )
    return args


def _tensor_tree_equal(actual, expected):
    import torch

    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or not torch.equal(actual, expected):
            raise AssertionError("Ported model output differs from reference output")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("Ported model output keys differ")
        for key in expected:
            _tensor_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError("Ported model output tuple length differs")
        for item, ref in zip(actual, expected):
            _tensor_tree_equal(item, ref)
    elif actual != expected:
        raise AssertionError("Ported model scalar output differs")


def _prepare_reviewed_sample(dataset, reference_sample, loaded, base):
    import numpy as np
    import torch
    from fisheye.endpoints.frame_store import (
        BoundedFrameStore,
        RepresentationDescriptor,
    )
    from fisheye.endpoints.preprocessing import CropStackPreprocessor
    from fisheye.endpoints.temporal import TrackTemporalSelector
    from fisheye.endpoints.types import EndpointQuery

    row = reference_sample["row"]
    frames = dataset.source._get_track_frames(row)
    center = int(row["frame_index"])
    offsets = loaded.preprocessing["temporal_offsets"]
    selection = TrackTemporalSelector(offsets=offsets).resolve(
        tuple(range(len(frames))), center
    )
    clip_id = str(row["location"]) + "/" + str(row["clip_name"])
    representation = loaded.preprocessing["representation"]
    with BoundedFrameStore(
        clip_id,
        ram_limit_bytes=64 * 1024**2,
        representations=(RepresentationDescriptor(representation),),
    ) as store:
        for index in set(selection.frame_ids):
            frame = frames[index]
            name = frame["file_name"] if isinstance(frame, dict) else frame
            with dataset.source._load_source_frame({**row, "file_name": name}) as image:
                store.put(index, np.asarray(image, dtype=np.uint8), representation)
        h, w = store.frame_hw
        # Explicit evaluator crop bounds remove dataset provenance assumptions
        # from this comparison; normal pipeline queries derive bounds from boxes.
        crop = base.clamp_bbox_xyxy(row["bbox_xyxy_padded"], w, h)
        query = EndpointQuery(
            query_id=f'{clip_id}:{row["track_id"]}:{row["frame_index"]}',
            clip_id=clip_id,
            frame_id=center,
            track_id=int(row["track_id"]),
            detection_id="reviewed",
            bbox_xyxy_px=tuple(row["bbox_xyxy_original"]),
            context_frame_ids=selection.frame_ids,
            context_repeated=selection.repeated,
            crop_xyxy_px=tuple(crop),
        )
        batch = next(
            CropStackPreprocessor(loaded.preprocessing).prepare(store, [query])
        )
    if not torch.equal(batch.images[0], reference_sample["img"]):
        difference = float((batch.images[0] - reference_sample["img"]).abs().max())
        raise AssertionError(f"Prepared image tensors differ (max error {difference})")
    if not torch.equal(batch.samples[0].target_box, reference_sample["target_box"]):
        raise AssertionError("Prepared target boxes differ")
    if "support" in reference_sample and not torch.equal(
        batch.support[0], reference_sample["support"]
    ):
        raise AssertionError("Prepared support masks differ")
    return batch


def _check_model(name, checkpoint, bundle, args, base, experiments, experiment_models):
    import torch
    from fisheye.endpoints.bundle import load_bundle, sha256_file

    source = checkpoint if checkpoint.is_absolute() else args.research_root / checkpoint
    loaded = load_bundle(bundle, device=args.device)
    if sha256_file(source) != loaded.spec["source"]["checkpoint_sha256"]:
        raise ValueError(
            f"{name}: research checkpoint differs from exported source hash"
        )
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    cfg = source_payload["config"]
    mode = cfg.get("experiment", {}).get("mode")
    model_loader = experiment_models.load_checkpoint if mode else base.load_checkpoint
    reference, _ = model_loader(source, args.device)
    reference.eval()
    dataset_class = experiments.ExperimentDataset if mode else base.EndpointDataset
    dataset = dataset_class(cfg, args.role)
    if args.queries > len(dataset):
        raise ValueError("Requested parity queries exceed reviewed dataset size")
    # Fixed, evenly spaced indexes are selected before inference; they do not
    # depend on model predictions, error, fish length, or checkpoint ranking.
    indices = [
        i * (len(dataset) - 1) // (args.queries - 1) for i in range(args.queries)
    ]
    rows = []
    for index in indices:
        sample = dataset[index]
        batch = _prepare_reviewed_sample(dataset, sample, loaded, base)
        with torch.inference_mode():
            reference_output = reference(sample["img"][None].to(args.device))
            actual_output = loaded.backend.predict(batch)
            raw_actual = (
                actual_output.prediction
                if loaded.spec["model"]["backend"] == "yolo_pose"
                else actual_output
            )
            _tensor_tree_equal(raw_actual, reference_output)
            if mode == "target_heatmap":
                points, entropy = reference.endpoints(
                    reference_output, sample["support"][None].to(args.device)
                )
                fields = experiments.heatmap_fields(points[0], entropy[0], sample)
            else:
                candidates = base.decode(
                    reference,
                    reference_output,
                    cfg["evaluation"]["candidate_confidence"],
                )[0]
                chosen = base.select_detection(candidates, sample["target_box"])
                fields = base.prediction_fields(
                    chosen, sample, cfg["evaluation"]["confidence_threshold"]
                )
            actual = loaded.decoder.decode(actual_output, batch)[0]
        expected_points = fields.get("global_endpoints_xyxy")
        if expected_points is None:
            if actual.status != "no_prediction" or actual.endpoints_xy_px is not None:
                raise AssertionError("Ported/reference prediction availability differs")
            coordinate_error = None
        else:
            expected = torch.tensor(expected_points).reshape(2, 2)
            got = torch.tensor(actual.endpoints_xy_px)
            if actual.status != "ok" or not torch.equal(got, expected):
                raise AssertionError("Ported/reference full-frame endpoints differ")
            coordinate_error = float((got - expected).abs().max())
        row = sample["row"]
        rows.append(
            {
                "dataset_index": index,
                "location": row["location"],
                "clip_name": row["clip_name"],
                "track_id": int(row["track_id"]),
                "track_frame_index": int(row["frame_index"]),
                "file_name": row["file_name"],
                "status": actual.status,
                "prepared_tensor_equal": True,
                "target_box_equal": True,
                "support_equal": True if "support" in sample else None,
                "all_raw_output_tensors_equal": True,
                "max_endpoint_error_px": coordinate_error,
                "endpoints_xy_px": actual.endpoints_xy_px,
            }
        )
    return {
        "backend": loaded.spec["model"]["backend"],
        "source_checkpoint": (
            SELECTED[name]
            if name in SELECTED and str(checkpoint) == SELECTED[name]
            else str(checkpoint)
        ),
        "source_checkpoint_sha256": loaded.spec["source"]["checkpoint_sha256"],
        "portable_checkpoint_sha256": loaded.spec["model"]["checkpoint_sha256"],
        "role": args.role,
        "available_queries": len(dataset),
        "queries": rows,
    }


def main(argv=None):
    args = _arguments(argv)
    args.research_root = args.research_root.resolve()
    if not (args.research_root / "model/yolo_pose/pipeline.py").is_file():
        raise ValueError(
            "--research-root must identify the maintained length_annotator checkout"
        )
    # This is deliberately confined to an opt-in development command; runtime
    # modules contain neither this path modification nor research imports.
    sys.path.insert(0, str(args.research_root))
    import cv2  # noqa: F401 -- load before Pillow/torch for Conda JPEG compatibility
    import torch

    torch.set_num_threads(args.threads)
    base = importlib.import_module("model.yolo_pose.pipeline")
    experiments = importlib.import_module("model.yolo_pose.experiments")
    experiment_models = importlib.import_module("model.yolo_pose.experiment_models")
    report = {
        "schema_version": 1,
        "reference_revision": "e29d4614ce523f483027bc4a61b8cb5993fade11",
        "selection": "Fixed evenly spaced reviewed validation indexes; checkpoints preselected using validation metrics",
        "scope": "Identical historical reviewed images and explicit evaluator crops; "
        "source-image equivalence with newly generated FishEye BGS is a separate check",
        "device": args.device,
        "models": {},
    }
    selected = {
        "yolo_5f": (args.yolo_checkpoint, args.yolo_bundle),
        "heatmap_5f": (args.heatmap_checkpoint, args.heatmap_bundle),
        "unblurred_signed_bgs_1f": (args.signed_checkpoint, args.signed_bundle),
    }
    for name in args.models:
        checkpoint, bundle = selected[name]
        report["models"][name] = _check_model(
            name, checkpoint, bundle, args, base, experiments, experiment_models
        )
        print(
            f"{name}: {args.queries} reviewed {args.role} queries match exactly",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

# Endpoint inference

FishEye contains the endpoint runtime, preprocessing, and model definitions.
It does not import `length_annotator` or require its data/manifests. Runtime code
is in `fisheye/endpoints/`; port provenance is in `vendor/PROVENANCE.md`.

## Enable the pipeline

### Shared unblurred signed BGS, one frame

The `unblurred_signed_bgs_1f` option copies **channel 1** from the existing
unblurred detector image before detector resizing. It retains one native-resolution
uint8 grayscale image, reuses the detector's background and signed residual
normalization, and performs no additional background subtraction. The fixed
background is shared for the entire clip. Negative residuals remain encoded below
the neutral value of 127 instead of being clipped to zero.

The signed-BGS group selects **seed 2**, using its validation-selected
`best_percentage_length_error.pt`. Among the five signed-BGS seeds it had the
lowest validation percentage length error (10.2437%) with 99.8346% availability.
Its observed test error is 11.2310% with 99.9898% availability; test scores did
not select this seed. The five-seed study still reports all seeds.

The local bundle is `weights/endpoints/unblurred_signed_bgs_1f_seed2`. Copy the
whole directory to deploy elsewhere. To recreate it from the trusted checkpoint:

```bash
python -m fisheye.endpoints.bundle export \
  /mnt/data/length_annotator/experiments/fisheye_inputs/unblurred_signed_bgs_center_pose_1f_bs16_cos100_seed2/best_percentage_length_error.pt \
  weights/endpoints/unblurred_signed_bgs_1f_seed2 \
  --profile configs/endpoints/unblurred_signed_bgs_1f_profile.json
```

Run the normal FishEye pipeline with that bundle:

```bash
python -m fisheye.main +endpoints=unblurred_signed_bgs_1f \
  input_path=/path/to/recording.aris output_dir=/path/to/output \
  platform.model.weights=weights/cfc_detect_yolov5s_v1.pt
```

This group sets `use_blur=false`, `only_pos_bgs=false` and background subtraction
on, and defaults to CPU endpoint inference plus the normal CSV/FC/XML exports. Set
`platform.inference.endpoints.device=cuda:0` to use CUDA for endpoints; detector
device/platform settings remain separately configurable. Existing detector
channels are retained for detection. The endpoint branch stores only their
center channel. Incompatible blur, sign, unwarped-coordinate or disabled-BGS
settings fail explicitly instead of silently changing the length inputs.

Edit **`configs/endpoints/unblurred_signed_bgs_1f.yaml`** under
`platform.inference.endpoints.track_length` to tune the length rules:

```yaml
track_length:
  enabled: true
  edge_margin_px: 25
  min_track_points: 5
  percentile: 50
```

1. Reject an observation if either endpoint is at or within `edge_margin_px`
   of either **sonar cone side**. Measure the shortest perpendicular distance
   in native warped-image pixels to the two calibrated outer beam boundaries.
   The sides come from the recording's `beam_width_data` and metric image
   origin/scale; they follow the cone as it narrows toward the sonar. This is
   not a distance to the rectangular image, detection box or model crop.
   No 25 px buffer is applied to the near/far range arcs or the top/bottom of
   the image. Endpoints outside the image or outside the cone sides are
   rejected independently. A zero margin still rejects endpoints on the cone
   sides. Missing/invalid calibration fails explicitly, without falling back
   to image edges. No new model confidence threshold is added.
2. Require at least `min_track_points` usable, matched observations **after**
   filtering. Failed predictions, invalid/nonpositive lengths and measurements
   without physical scale do not count. Tracks below this minimum have no length.
   These rules affect measurement availability; fish counting remains independent.
3. Compute the configured percentile of **all remaining observation lengths**
   in each track, in centimetres. `50` is the median. The range is `0..100`,
   with NumPy's linear interpolation (including even-sized medians). No other
   smoothing, time-window selection or legacy length filtering is applied.

For per-run overrides:

```bash
python -m fisheye.main +endpoints=unblurred_signed_bgs_1f \
  input_path=/path/to/recording.aris output_dir=/path/to/output \
  platform.model.weights=weights/cfc_detect_yolov5s_v1.pt \
  platform.inference.endpoints.track_length.edge_margin_px=25 \
  platform.inference.endpoints.track_length.min_track_points=5 \
  platform.inference.endpoints.track_length.percentile=50
```

`recording.endpoints.json` retains raw per-observation results for inspection,
with a `length_filter` decision showing eligibility, rejection reason and
whether the observation entered an accepted track length. `track_lengths`
contains accepted lengths and explicit discarded-track reasons/counts, using
`null` for missing lengths. The resolved rules, calibrated cone geometry and model hashes are recorded.
`nearest_cone_side_px` records each prediction's signed perpendicular distance;
a negative value identifies an endpoint outside a cone side.
Rejected raw observations are diagnostic records and never enter aggregation.
Set `export_options=[]` for JSON only, or `track_length.enabled=false` to disable
aggregation and keep raw endpoint inference.

Accepted aggregates also populate `L(cm)` in normal detailed CSV, FC and XML
outputs. The existing external missing-length convention is `0`; JSON retains
`null` and a rejection reason. For formats requiring endpoints, the closest
usable observation supplies a real representative frame/endpoint pair (ties
choose the earlier frame). This segment's measured length can differ from an
interpolated percentile. JSON retains both values; detailed CSV and XML identify
the aggregation method, percentile, usable-point count and representative length.
`Frame#` identifies that representative; `Crossing Frame#` / XML
`CrossingFrameIndex` preserves the original count crossing. The percentile is
never used to fabricate endpoint coordinates.

In `FCe_*.txt`, `L(cm)` contains the configured percentile length. The final
`Comment` (notes) column also records `length_percentile` and a JSON mapping
`length_percentiles_cm` with keys `"0"`, `"10"`, …, `"100"`. These eleven values
are lengths in centimetres, rounded to two decimal places, calculated from the
same filtered observations using linear interpolation. For example,
`length_percentile=75; length_percentiles_cm={"0":20.0,"10":25.0,...,"100":70.0}`.
This allows later selection of another recorded percentile using the FCe file.
Existing comments and the app version are retained. Tracks below the minimum
usable-point count have `L(cm)=0` and no percentile table. The endpoint JSON also
retains the table at full precision in each accepted `track_lengths` entry.

Endpoint runs process recordings even when an old FC count file exists and
replace endpoint JSON atomically. Normal exports are now requested by default;
use `export_options=[]` when preserving existing count exports is desired.

**Native source resolution and model input size are distinct:** the current
single-frame experiments use crops letterboxed to 640×640. This deployment keeps
full-resolution source pixels, applies the checkpoint's trained crop/letterbox
transform, and maps predictions back to the native image. It does not silently
switch a 640-trained model to native-size inference.

Recheck seed 2 against the research evaluator with:

```bash
python -m scripts.check_endpoint_reference_parity \
  --research-root /path/to/length_annotator --models unblurred_signed_bgs_1f \
  --signed-checkpoint /mnt/data/length_annotator/experiments/fisheye_inputs/unblurred_signed_bgs_center_pose_1f_bs16_cos100_seed2/best_percentage_length_error.pt \
  --signed-bundle weights/endpoints/unblurred_signed_bgs_1f_seed2 \
  --queries 30 --output /tmp/signed-bgs-parity.json
```

## Runtime and resources

Install the project dependencies (`poetry install`). Endpoint YOLO models use
`ultralytics==8.4.51`; bundle loading checks that version. The bundle owns the
backend, decoder, and trained preprocessing. Runtime configuration exposes only
bundle location, device, crop batch size, frame-store limits, prediction export,
and track-length rules. Inference uses float32. To change trained preprocessing,
export and qualify a new bundle using an explicit export profile.

Legacy YOLOv5 detector checkpoints contain full Python model objects. FishEye
keeps their trusted-checkpoint loading behavior inside the YOLOv5 loader when
using newer Ultralytics versions, without changing global `torch.load` behavior.
Endpoint bundle weights continue to load with `weights_only=True`. If an import
reports that `yolov5.load` is missing, check that the installed `yolov5` package
is being imported rather than a local directory with the same name.

Prediction runs after tracking and uses the actual matched detection boxes.
Source frame IDs are retained across partial batches and reverse tracking.
Only matched observations from retained tracks receive endpoint predictions.

Retained native images have bounded RAM usage; excess images spill losslessly
to the configured scratch directory. Scratch images are removed when a clip
finishes or fails. `frame_store.ram_limit_mb` defaults to 512 and
`frame_store.disk_limit_gb` to 20; `frame_store.directory` selects the scratch
parent. During capture, `cache_bg_frames` is disabled with a warning to avoid
another unbounded copy. Limits cover retained pixels, not background statistics,
preprocessing arrays, loader prefetch, or model memory.

A flat initial background sample uses a fixed normalization of 255, shared with
the detector. This preserves foreground arriving later in the recording. Output
records the sampled and effective normalization, source frame IDs, and background
fingerprint. No separate endpoint background estimate or positive-only image
branch is computed.

Unexpected endpoint errors fail the clip and processing continues with the
remaining clips. Successful clips receive their requested combined summary;
the job then raises `EndpointBatchError` (a nonzero CLI exit). Its `results` and
`failures` attributes remain available to Python callers. An atomic
`endpoint_failures_<unique-id>.json` records failed source paths and errors in
the output directory, or beside failed recordings when no output directory was
specified. Failed clips never become successful zero-count rows.

## Bundles and verification

Transfer the whole bundle directory (`inference_spec.json` and `weights.pt`)
to deploy. Runtime loading uses tensor state and embedded architecture without
research training code or pretrained-weight downloads. Validate a copied bundle:

```bash
python -m fisheye.endpoints.bundle validate \
  weights/endpoints/unblurred_signed_bgs_1f_seed2
```

The signed one-frame group is the supported FishEye integration. The former
`yolo_5f` and `heatmap_5f` deployment groups and positive-BGS export profiles have
been removed. Older model adapters remain available for independent research
parity checks; positive-BGS bundles cannot be used with the FishEye dataloader.
The regular dataloader's existing `only_pos_bgs` option remains available for
other detector workflows; endpoint capture requires it to be false.

Tests cover native detector pixel parity, one shared background calculation,
source identities, detection associations, bundle compatibility, aggregation,
exports, bounded storage, and cleanup after failures. Hardware throughput and
accuracy on freshly captured FishEye images remain deployment measurements.

Seed-2 model qualification is recorded in
[signed_bgs_seed2_parity.json](signed_bgs_seed2_parity.json): 30 fixed validation
queries compare prepared tensors, raw outputs, availability, and decoded global
endpoints against the research checkpoint. The earlier
[endpoint_model_parity.json](endpoint_model_parity.json) is historical evidence
for the five-frame adapters.

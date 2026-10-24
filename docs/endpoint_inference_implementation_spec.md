# Endpoint inference integration specification

Status: archived design proposal, 2026-10-20. Superseded by [the endpoint guide](endpoints.md).

The deployed pipeline now uses signed BGS copied from detector channel 1 and
one-frame seed-2 pose inference, followed by cone-side filtering and track length
aggregation. The independent positive-BGS branch, five-frame deployment groups,
and runtime component/profile overrides described below have been removed.
This document records the original design, not current configuration requirements.

## Confirmed deployment choices

- All runtime code needed for endpoint inference lives in the FishEye repository.
  FishEye must run without this repository installed or present on disk. Port or
  copy current, validated model/preprocessing code as needed, preserving provenance
  and any license requirements. Normal third-party library dependencies remain.
- Initial implementation supports both current best five-frame YOLO keypoint and
  five-frame heatmap methods. Audit available comparison reports to identify the
  exact checkpoint/configuration/hash for each; do not assume a heatmap architecture
  or declare a winner from filenames. Both families are initial deliverables.
- Endpoint input comes from original-resolution warped images with unblurred
  background subtraction. Preserve the detector's independently configured inputs.
- Small source-image differences from historical training data are acceptable to
  investigate and measure. They do not block integration by themselves. Generate
  subsequent training images using the exact FishEye endpoint preprocessing path,
  record its configuration/version, and retrain/qualify the models on those inputs.
- Exact tensor/decoder parity on identical supplied images remains required to
  validate copied runtime code. This is distinct from source-image equivalence
  between the historical dataset and fresh FishEye processing.

## Purpose and decisions

Add endpoint inference to FishEye using frames already decoded for detection and
the boxes associated with tracked fish. Return individual endpoint predictions
with enough provenance and diagnostic information for filtering later.

The initial integration runs after tracking, within the existing offline clip
pipeline. A persistent predictor composes independently replaceable preprocessing,
model, and decoding components. A separate query builder owns detection association
and supplies chronological track observations; a configurable temporal selector
owns frame selection. A frame store retains the image representations requested
by the selected preprocessor during the detection pass.

The predictor also accepts single-frame queries when the checkpoint was trained
for single-frame input. Supporting these queries does not require an initial
before-tracking integration or a streaming scheduler.

Use current model implementations and checkpoint loaders from `length_annotator`
as the behavioral reference. Treat FishEye's existing `lengths/` code and tests
as historical integration hints, not a validated implementation or contract.

## Scope

The implementation includes:

- A portable inference bundle containing weights and a versioned inference spec.
- Independently replaceable image preprocessing, temporal selection/stacking,
  model backends, and output decoders behind explicit compatibility contracts.
- A predictor accepting in-memory images through a frame-store interface and
  explicit endpoint queries, without training manifests or reviewed labels.
- An endpoint-specific image branch during FishEye frame preprocessing.
- Detection association, temporal query construction, and bounded frame storage.
- Per-query endpoint results, raw scores, diagnostic flags, and clip-level export.
- Parity checks against existing model evaluation and FishEye integration tests.

Track-level length aggregation, measurement acceptance thresholds, temporal
outlier filtering, model retraining, model ranking, live streaming, and model
runtime conversion are outside the initial implementation. Existing detection,
tracking, counting, and crossing export behavior remain the baseline to compare
against when endpoint inference is enabled or disabled.

## Existing integration points

Inspected source repository: `/home/mahobley/Code/fisheye`.

| Source | Observed behavior | Implementation implication |
| --- | --- | --- |
| `fisheye/pipelines/detection.py`, `ObjectDetectionPipeline._run_inference` | Iterates decoded batches, runs detection and NMS, then optionally calls the old length estimator | Capture endpoint images and frame/detection identities here; replace the legacy inference hook for the new path |
| `fisheye/pipelines/pipeline.py`, `DetectTrackCountPipeline._run` | Completes detection, calls `run_tracker`, then processes old length estimates | Invoke query building and endpoint prediction after tracking |
| `fisheye/boxes.py`, `NMSProcessor` | Low/high NMS outputs pass through `normalize_boxes_for_tracking` | These boxes are normalized `xyxy`; convert explicitly to original-image pixels |
| `fisheye/track/tracker.py` | Tracks include frame numbers and `bbox_index`/`det_index` associations; finalization can filter tracks | Validate associations and frame mapping; use retained tracks for initial endpoint queries |
| `fisheye/dataloaders/base.py` | Builds raw/BGS/temporal-difference channels; `only_pos_bgs=False` shifts signed BGS | Generate an independent positive-only endpoint image before the display-range shift |
| `fisheye/dataloaders/yolo.py` | Returns original-resolution processed images optionally; these are float tensors in the 0–255 range | Original resolution alone does not establish endpoint input compatibility |

Do not infer frame identity solely from `batch_idx * configured_batch_size`.
Propagate the actual source frame number through loading, detection, and tracking.
Confirm the loader's endpoint conventions with fixtures, including the final
partial batch, rather than deriving identity from undocumented index behavior.

## Architecture and ownership

```text
Decoded warped sonar frames
  ├─ existing detector preprocessing → detector → NMS → detection records
  └─ configured image preparation → CPU/disk frame store
                                                   │
Detection records → tracker → retained tracks       │
                                │                  │
                    query builder + temporal selector
                                └──────┬───────────┘
                  predictor: preprocessor → model → decoder
                                     │
                         per-query endpoint results
                                     │
                        export / future filtering
```

Suggested code locations, subject to existing package conventions:

| Location | Responsibility |
| --- | --- |
| `fisheye/endpoints/predictor.py` | Public predictor and backend dispatch |
| `fisheye/endpoints/types.py` | Inference spec, query/result types, coordinate transforms |
| `fisheye/endpoints/preprocessing/` | Replaceable image preparation, temporal selectors/stackers, crop and tensor transforms |
| `fisheye/endpoints/backends/` | Replaceable model loading and forward execution, including ported model definitions |
| `fisheye/endpoints/decoders/` | Pose/heatmap decoding into the shared endpoint result contract |
| `fisheye/endpoints/export_bundle.py` | Resolve and export the exact checkpoint preprocessing/decoding contract |
| `fisheye/endpoints/frame_store.py` | Clip-scoped frame retention and cleanup |
| `fisheye/endpoints/queries.py` | Detection association and chronological track queries |
| `fisheye/endpoints/processor.py` | Run prediction for a clip and export results |
| `fisheye/configs/inference.py` | FishEye runtime endpoint configuration |

Port runtime code into FishEye's existing package; do not introduce a separately
installed dependency on this research repository, working-directory-dependent
imports, or `sys.path` modifications. Record source revision and parity fixtures
for copied code. Use maintained model definitions from this repository rather
than FishEye's old length models. Verify loading in an environment where this
repository is absent, including checkpoint class/import-path compatibility.

Create the predictor once per pipeline worker, reuse it across clips, and keep
clip metadata and frames outside its long-lived model state. Concurrent calls
to a shared predictor are unsupported initially; queue batches through its owner.

## Component contracts and interchangeability

Swapping preprocessing or model families must require configuration or dependency
injection only, without changes to FishEye detection, tracking, or result consumers.
Use typed Python protocols and a small explicit component registry. Components
have a versioned identifier and serializable configuration. Programmatic callers
may inject implementations directly using the same protocols.

Separate these responsibilities:

| Component | Input → output | Owns |
| --- | --- | --- |
| `ImagePreparation` | Decoded frames + clip background metadata → named stored images | BGS/raw representation, blur, clipping, quantization; declares source data requirements |
| `TemporalSelector` | Center observation + chronological track/frame index → context IDs and repetition markers | Single frame, symmetric neighbors, or other explicit selection policies |
| `InputPreprocessor` | Frame store + resolved queries → prepared model inputs | Crop geometry, temporal stacking/order, normalization, resizing, optional masks, batching and inverse-transform metadata |
| `ModelBackend` | Prepared tensor batch → backend output | Checkpoint loading, input requirements, device/precision and forward pass |
| `OutputDecoder` | Backend output + prepared-sample metadata → endpoint results | Pose matching or heatmap decoding, inverse coordinates, scores and diagnostics |

`InputPreprocessor` composes replaceable temporal stacking and spatial/tensor
transforms. It must support replacing the whole preparation pipeline as well as
changing an individual policy. Temporal selection happens before pixel fetching;
stacking the selected frames happens inside input preparation. Neither operation
belongs inside a YOLO or heatmap model class.

Temporal length is configurable rather than hardcoded to five. Support 1, 3, 5,
and 9 total frames through the same selector, stacker, query, and prepared-batch
contracts. For symmetric context, configure a nonnegative integer radius `r`:
offsets are `[-r, ..., 0, ..., r]` and total frames are `2*r + 1`. Radii 0, 1, 2,
and 4 therefore produce 1, 3, 5, and 9 frames. Also allow an explicit ordered
offset list for future asymmetric policies; record its sampling basis (track
positions or source-frame offsets), ordering, and boundary behavior in the profile.
Reject ambiguous configurations that specify inconsistent radii and offsets.

Derive input channel counts from the selected frames, per-frame representations,
and any auxiliary mask channels. Model construction/loading must use the matching
checkpoint input specification, with no fixed five-channel assumptions in shared
runtime code. Changing temporal length selects a compatible trained checkpoint;
it does not adapt existing weights automatically. The first release qualifies
the two five-frame models, while lightweight tests cover 1/3/5/9-frame selection,
stacking, boundary repetition, and compatibility rejection for mismatched models.

Conceptual interfaces (names may follow final package conventions):

```python
selector.resolve(observation, track_index, frame_index) -> ResolvedContext
preprocessor.prepare(frame_store, queries) -> Iterable[PreparedBatch]
backend.predict(batch: PreparedBatch) -> BackendOutput
decoder.decode(output: BackendOutput, batch: PreparedBatch) -> list[EndpointResult]

predictor = EndpointPredictor(
    preprocessor=preprocessor,
    backend=backend,
    decoder=decoder,
)
```

`PreparedBatch` carries the input tensor(s), ordered query IDs, target boxes in
model coordinates, real-image support, per-sample geometry, and input descriptor.
That descriptor specifies tensor layout, channel semantics/order, value range,
dtype, spatial constraints, and optional auxiliary inputs. It supports the current
`B×C×H×W` models without assuming all future models consume a channel stack.
The preprocessor owns collation because padding/canvas changes can affect results.
The backend must not silently resize, normalize, stack frames, or add masks.
The decoder must not fetch images or reconstruct undocumented crop transforms.

YOLO keypoints and heatmap models expose the same public prediction API and
`EndpointResult` schema, while preserving their own backend outputs and score
semantics. Switching model families normally changes the backend and its compatible
decoder, and may select a different preprocessing profile. A native U-Net need
not accept YOLO letterboxing to satisfy interchangeability.

Validate component compatibility at initialization, before processing a clip:

- Available stored representations satisfy image preparation requirements.
- Temporal selection and stacking produce the required channel count/order and
  source image semantics for the checkpoint.
- Prepared inputs match model layout, shape/stride, range and mask requirements.
- Decoder supports the backend output schema and coordinate convention.

Modularity does not imply arbitrary combinations are valid: a five-channel
checkpoint cannot consume a single-frame tensor, and a heatmap decoder cannot
consume pose candidates. Reject incompatible combinations with an actionable
error. Alternative preprocessing for a compatible checkpoint can be selected as
a versioned experiment profile; record the deviation from its reference profile
and qualify its accuracy before production use.

Provide reference profiles for single-frame and track-temporal stacking, native
U-Net heatmaps, YOLO keypoints, and the selected YOLO-backed heatmap option as
their adapters are implemented. Adding a component must not require modifying
the predictor orchestration loop. Each prediction export records resolved
component IDs, versions, configurations, and model hash.

## Portable inference bundle

Each bundle contains a checkpoint and `inference_spec.json`. Record:

- Schema version, model backend/architecture, checkpoint hash, and code version.
- Required runtime versions, including the custom YOLO implementation dependency.
- Image preprocessing identifier and resolved parameters: warped coordinate
  space, background statistics, blur, positive-only clipping, quantization,
  source value range, and conversion to `[0, 1]`.
- Temporal offsets and ordering, track-position selection, and boundary policy.
- Crop expansion and integer rounding policy.
- Resize/interpolation/padding settings, mask conditioning if applicable, and
  native batching/canvas policy.
- Endpoint decoder and its parameters; pose candidate decoding and matching rules.
- Endpoint ordering and meanings of exported scores.
- Component IDs/versions and their input/output compatibility descriptors,
  including the reference preprocessing profile and compatible decoder.

Export resolves these fields from the saved configuration and the current
evaluation implementation. Some historical settings, such as the saved crop
policy, may live in dataset provenance rather than the checkpoint. Such fields
must be supplied explicitly and recorded during export. Fail export if required
semantics cannot be resolved; do not silently guess from a filename.

Runtime options control device, crop batch size, frame-store capacity/location,
and output paths. Configuration can select separate model and preprocessing
profiles. Resolve these into a complete inference spec and validate compatibility
before use; preprocessing and decoder changes create a distinct recorded spec
and require qualification. A checkpoint trained for one frame cannot be switched
to five frames merely by selecting a temporal profile.

## Data contracts

Use typed records internally; export JSON-compatible values.

### Frame store

```python
class FrameStore:
    def get(self, frame_id: int, representation: str) -> np.ndarray:
        # Array layout/dtype/range described by the named representation
        ...
```

The initial reference representation is `bgs_positive_u8`: H×W uint8 in original
warped pixels. Other preprocessors may request raw images or other representations;
declare these requirements before loading starts. Retain only required
representations, and never silently substitute BGS for raw pixels. If a new
profile needs unavailable source data, require a new capture pass.

Each store is scoped to one clip. Its metadata includes clip identity, frame
dimensions, representation descriptors, preprocessing fingerprints, and available
source frame IDs. All
temporal images in a query must have identical dimensions and preprocessing.
Stored images are immutable. CPU/disk access must not allocate full-frame GPU
tensors. Fetch, crop, then convert the cropped pixels to float.

### Detection record

Preserve `clip_id`, `frame_id`, stable `detection_id`, detector score, source
group, original group index, and `bbox_xyxy_px`. Keep explicit mappings from the
legacy low/high dictionary keys to these records while the tracker uses them.
The same detection appearing in both groups must not cause duplicate endpoint
inference. Do not assume the two NMS outputs are identical subsets; only reuse
results when query identity, box, context frames, and preprocessing match.

### Endpoint query

```python
@dataclass(frozen=True)
class EndpointQuery:
    query_id: str
    clip_id: str
    frame_id: int
    track_id: int | None
    detection_id: str | None
    bbox_xyxy_px: tuple[float, float, float, float]
    context_frame_ids: tuple[int, ...]
    context_repeated: tuple[bool, ...]
    crop_xyxy_px: tuple[int, int, int, int] | None = None
```

`context_repeated` identifies positions filled by boundary repetition, rather
than ordinary observations. An explicit crop is supported for evaluator parity;
normal production queries derive it from the input box and inference spec.

The lower-level predictor requires explicit context IDs and therefore does not
depend on tracker types. A convenience single-frame caller supplies `(frame_id,)`.

### Endpoint result

Return exactly one result per query, preserving input order and `query_id`:

- Input clip/frame/track/detection identity and target box.
- `status`: `ok`, `no_prediction`, `invalid_query`, `missing_frame`, or
  `inference_error`, plus a structured reason where applicable.
- `endpoints_xy_px`: two floating-point full-frame points, or null.
- `length_pixels`: Euclidean endpoint distance, or null.
- Optional `length_cm` when a valid `pixel_meter_size` is supplied. This is an
  unfiltered per-frame measurement: `length_pixels * pixel_meter_size * 100`.
- Backend-specific `scores` with names and documented semantics; no universal
  calibrated confidence or acceptance boolean.
- Context frame IDs and repetition markers; crop bounds and coordinate transform.
- Diagnostic flags, including endpoints outside the real crop/frame when present.
- Bundle/model identity, referenced through shared clip metadata if desired.

Null means unavailable. Never substitute zero endpoints or zero length for a
miss. `ok` describes an available, finite prediction, not a reliable measurement.
Order output points by increasing x, stably on ties, without assigning anatomical
head/tail identity. Reorder endpoint-specific scores with their points.

## Query construction and temporal context

1. Build an explicit mapping between tracker frame numbers and source frame IDs.
   Include frames without detections. Verify reverse tracking does not reverse
   the chronological context presented to the model.
2. Recover each track observation's matched source detection using its validated
   group/index association. Use the original detection box for the target query;
   tracker box smoothing must not silently change crop geometry.
3. Build a chronological list of matched observations for each retained track.
   Tracker-only extrapolated observations are not endpoint query centers or
   temporal neighbors in the initial policy. Record skipped observations and
   validate that this policy is compatible with reference track-frame lists.
4. Pass each center and its chronological track index to the configured temporal
   selector. The reference five-frame selector uses track positions
   `[-2, -1, 0, 1, 2]`; frame numbers need not be consecutive. The single-frame
   selector returns only the center. Other selectors are separate versioned policies.
5. The reference selector clamps positions at actual track boundaries and records
   repetition. Do not
   clamp missing/corrupt interior images or cross tracks/clips.

Missing associations must produce a diagnostic, rather than falling back to an
arbitrary detection. FishEye's association logic is not assumed correct because
fields exist: verify it with multiple-fish fixtures and both tracking directions.
If reference frame lists include a different observation policy, resolve and
version that difference before enabling the five-frame production path.

The initial scope predicts retained tracks after existing tracker finalization.
It does not provide predictions for detections removed by tracking or the
tracker's existing box-size/travel filters. Record those selection settings in
clip metadata so downstream users understand output coverage.

## Image preprocessing and crop geometry

### Source inspection: original resolution and unblurred BGS

The current FishEye source already exposes the necessary controls:

- `YOLOARISBatchedDataset.load_image` copies `img_original` before detector resize
  when `return_original_image=True`. This preserves original warped image
  resolution; it is a processed image, not raw unwarped sonar samples.
- `BaseDataset._compute_bg_subtraction` forwards `use_blur` to background-statistic
  computation. `compute_bg_subtraction(..., use_blur=False)` computes the mean and
  normalization from unblurred frames.
- `BaseDataset._stack_preprocessed_channels` also bypasses Gaussian blur when
  `use_blur=False`. Thus both the background estimate and per-frame subtraction
  have an existing no-blur path.
- The resulting middle channel contains BGS. `only_pos_bgs=True` selects positive
  residuals before uint8 conversion; `False` shifts signed residuals instead.

These findings are source inspection, not an end-to-end runtime qualification.
Exercise actual clips to verify dimensions, frame alignment, both no-blur paths,
and numerical behavior (including a zero normalization range). The existing
dataset flags are shared with detector preprocessing. Use a separate endpoint
preparation profile with `use_blur=False`, rather than changing those shared flags
and unintentionally changing detector inputs. Capture endpoint images before
detector resizing; model-specific crop resizing remains configurable afterward.

Generate endpoint-compatible BGS from decoded warped frames and the appropriate
background statistics. Reuse intermediate blur/subtraction work where settings
match. Apply positive-only clipping before uint8 quantization for endpoint input.
Do not derive endpoint input from the already shifted signed-BGS display channel:
its clipping/quantization can lose information. Preserve the detector's configured
image branch independently.

Compare this branch against the processed images used for training. Saved JPEG
frames may differ from freshly generated uint8 pixels because of encoding or
historical preprocessing. Establish exact parity on identical supplied images
separately from end-to-end agreement on freshly decoded sonar; document any
remaining source-image difference and its measured prediction impact.

For production crops, reproduce the original annotation generator's integer
rounding and 100%-per-side box expansion, clipped to source boundaries. Extract
and test that algorithm rather than introducing an approximately equivalent
float formula. When supplied, reference saved crop bounds take precedence.

Use the center-frame crop at the same spatial coordinates in every temporal
image. Do not recenter each image using its neighbor's box. Reject degenerate
or nonfinite boxes. Disable training augmentations and TTA.

Backend adapters preserve their reference evaluator behavior:

| Backend | Required behavior |
| --- | --- |
| Native U-Net | Native crop resolution, matching canvas/collation policy and configured heatmap decoder; decode over the real crop without ground-truth endpoint windows |
| Original/center-only/all-box pose | Checkpoint-specific letterbox or native stride padding; decode candidates and match the requested target using the current target-box matching algorithm |
| Target-mask pose/heatmap | Add the target mask using existing rasterization/support and decoding semantics; disable training mask jitter |

All-box inference does not require neighboring annotation labels: they were
training supervision. It still needs the requested target box for selection.
Implement and qualify both selected five-frame production backends. Other bundle
backends fail explicitly until their adapters pass parity checks.

Retain actual rounded x/y resize gains, letterbox offsets, and crop origin.
Invert transforms using those exact values. For letterboxed outputs:

```text
x_full = (x_model - pad_left) / gain_x + crop_left
y_full = (y_model - pad_top)  / gain_y + crop_top
```

Do not silently clamp predicted endpoints to the target box or replace invalid
predictions with box corners. Preserve finite outside-crop predictions with
diagnostics where the reference decoder permits them; invalid numerical outputs
produce `no_prediction` with a reason.

## Runtime and storage

Initial runtime is the existing PyTorch backend with the exact model definition
and dependency versions required by the bundle. Set evaluation mode and disable
gradient tracking. Establish float32 parity first; enable mixed precision only
after measuring prediction differences for that backend.

Batch endpoint crops independently of detection batch size. Start parity runs
with batch size 1 and the reference canvas policy. Native-resolution batching
can alter padded canvases and predictions; larger or shape-grouped batches need
their own parity/quality check. Fixed-size letterbox crops can use ordinary
batching once verified. Batch size is configurable, with no unmeasured throughput
or GPU-memory promise.

The frame store has a configured RAM limit and spills uint8 endpoint images to
clip-scoped, lossless disk storage. Account for approximately `N * H * W` bytes
of uncompressed single-channel pixels, plus indexing and active batches. Enforce
disk limits and fail clearly if capacity is unavailable. Retain all emitted
frames during detection because final track membership is not yet available.
Prune unneeded frames after query construction if useful.

Always clean up temporary stores through an explicit context/finally path.
Stores cannot leak images into another clip. Persistent cache reuse is outside
the first version; each run creates a fresh store. Export endpoint results
atomically on successful completion.

## Configuration and FishEye wiring

Proposed runtime configuration, separate from legacy `length_config`:

```yaml
endpoints:
  enabled: false
  model:
    bundle: /path/to/endpoint_bundle
    backend: from_bundle
    decoder: from_bundle
  preprocessing:
    profile: from_bundle
    # A named versioned profile may replace the reference pipeline.
    # Its config includes image preparation, temporal selector/stacker,
    # crop, normalization, spatial transforms, and auxiliary inputs.
  device: cuda:0
  crop_batch_size: 1
  precision: float32
  frame_store:
    ram_limit_mb: 512
    disk_limit_gb: 20
    directory: /path/to/scratch
  failure_policy: fail_clip
  export_predictions: true
```

Capacity values are starting configuration examples, not sizing guarantees.
By default, resolve every component from the bundle's reference profile. Explicit
profile/backend/decoder selections are allowed after compatibility validation;
write their fully resolved inference spec to output provenance. Configure component
parameters through versioned profiles rather than scattering transform overrides
across the FishEye pipeline.

Update the pipeline builder to inject a predictor and endpoint processor. Enable
the endpoint image branch and frame-store capture only when requested. Avoid
running the legacy estimator in parallel with the new path. Keep detector return
contracts compatible where practical; add a named clip result/context to carry
frame-store and detection records rather than extending positional tuples again.

In `DetectTrackCountPipeline._run`, build queries and predict after tracking,
then write an endpoint sidecar keyed by stable query IDs. Keep crossing summaries
independent until a separate filtering/aggregation component is specified.
Do not feed raw per-frame estimates into the old `LengthProcessor` by default.

## Errors and observability

Bundle incompatibility, invalid preprocessing configuration, disk exhaustion,
CUDA errors, or systematic model failures fail the clip explicitly. Do not turn
them into empty successful endpoint output. Existing broad retry/default-return
wrappers must preserve an observable failed status; ensure retries release frame
stores and do not publish duplicate or partial successful sidecars.

Invalid individual queries and missing frames return structured per-query results.
No matching pose candidate is an expected `no_prediction`, not a process error.
Unexpected backend errors stop the clip under the initial `fail_clip` policy.

Log counts for detections, retained tracks, valid queries, skipped associations,
repeated context positions, available predictions, and each failure reason.
Measure frame capture/preprocessing, query construction, crop preparation,
transfer, model execution, decoding, and export times; report peak RAM, scratch
usage, and GPU memory during representative benchmarks. Include detector and
endpoint model hashes/configuration in exported provenance.

## Validation and acceptance

1. **Preprocessing:** Verify endpoint pixels against reference processing; confirm
   enabling endpoints does not change detector input tensors. Cover signed versus
   positive-only BGS, uint8 scaling, and immutable shared inputs.
2. **Geometry:** Test fractional boxes, rounding, image-edge crops, narrow crops,
   letterbox rounding, native padding, masks, and inverse coordinate transforms.
3. **Temporal/identity:** Test track boundaries, sparse frame numbers, clip
   isolation, reverse tracking, multiple fish, missing associations, empty
   frames, and the final partial detection batch.
4. **Model parity:** On identical supplied frames, boxes, and saved crop bounds,
   compare prepared tensors, decoded endpoints, scores, and missing-prediction
   decisions with the existing evaluator. Preserve its batch/canvas settings.
   Set documented numeric tolerances from the reference backend before accepting
   the adapter; require exact preprocessing equality where arithmetic is shared.
5. **Production inputs:** Separately evaluate freshly decoded sonar and detected
   boxes. Report endpoint/length errors and availability using reviewed labels
   where available. Reviewed-box parity does not establish detected-box quality.
6. **Integration:** Run a representative clip with endpoints disabled and enabled.
   Require unchanged detection/tracking/counting outputs, correct exported
   identities, one result per query, and no leakage across clips or batches.
7. **Resources/failures:** Exercise spill-to-disk, cleanup after failures, repeated
   clips, missing images, insufficient storage, and interrupted exports. Benchmark
   complete clips with detection sharing the intended GPU.
8. **Component interchangeability:** Using the same FishEye integration and result
   consumer, run a compatible YOLO keypoint bundle and a heatmap bundle by changing
   configuration only. Test five-frame preparation with both matching checkpoints
   and single-frame selection/stacking contracts with lightweight fixtures;
   replace a preprocessor through injection without changing model
   code. Reject channel, representation, shape, mask and decoder mismatches before
   clip execution. Verify component provenance and score/point ordering.

Acceptance requires parity for the chosen backend, validated frame/detection
association, bounded resource use, complete per-query accounting, and explicit
failure reporting. Choose deployment throughput and quality targets from the
measured representative workloads; this spec does not assume a hardware budget
or establish a winning model.

## Implementation sequence

1. Identify the current best five-frame YOLO keypoint and heatmap checkpoints from
   comparison reports; record configurations/hashes and export reference bundles.
   Establish FishEye runtime module ownership and identical-input parity fixtures.
2. Implement the stateless query/result and prepared-batch contracts, component
   protocols/registry, compatibility validation, and first reference preprocessing,
   backend and decoder; port and qualify both five-frame model families inside
   FishEye. Demonstrate swapping between them through configuration. Single-frame
   profiles remain an extension supported by the contracts, not a first-release
   checkpoint qualification requirement.
3. Add the independent endpoint image branch, explicit frame IDs, frame storage,
   and stable detection records to FishEye's detection pass.
4. Validate tracker associations and implement chronological query construction.
5. Wire the processor after tracking, add configuration and sidecar export, and
   exercise full clips and failure cleanup with endpoint inference off by default.
6. Validate detected-box quality, benchmark resource use, and qualify larger crop
   batches or mixed precision before enabling the chosen deployment settings.

Before implementation is complete, resolve the two exact checkpoint identities,
input preprocessing provenance and measured differences, tracker observation policy,
representative hardware/workload, and numeric parity tolerances. These are
qualification decisions within the sequence above, not reasons to block work on
the shared predictor and data contracts.

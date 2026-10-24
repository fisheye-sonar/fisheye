# Endpoint model provenance

The inference implementations in this directory were ported from
`length_annotator` revision `e29d4614ce523f483027bc4a61b8cb5993fade11`:

- `native_heatmap.py`: `model/head_tail_models.py` (unchanged model definitions).
- `target_heatmap.py`: inference-only `TargetHeatmapModel` from
  `model/yolo_pose/experiment_models.py`; training losses were omitted.
- `../decoders.py`: pose matching/NMS and target heatmap distribution from
  `model/yolo_pose/pipeline.py` and `experiment_models.py`, plus native
  `mapTokpt` from `model/head_tail_utils.py`.

The source repository did not contain a separate LICENSE file at export time.
This copy is made under the repository owner's explicit authorization. FishEye's
existing license and notices still apply to FishEye; third-party Ultralytics code
is installed separately and retains its own license. No research repository is
imported at runtime. Serialized bundles contain state dictionaries and primitive
configuration values, never research model class objects.

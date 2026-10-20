# FishEye

FishEye is a Python library for automated fish counting from imaging sonar data using computer vision. It automates core workflows such as data conversion, preprocessing, inference, and dataset creation, and provides pretrained models for fish detection.

---

## Installation & Setup

### Requirements

* **Python**: 3.10.14 (or 3.8 for NVIDIA Jetson)

  * Recommend using [pyenv](https://github.com/pyenv/pyenv) for Python and virtual environment management
* **Poetry**: [https://python-poetry.org/docs/](https://python-poetry.org/docs/) (dependency management & packaging)
* **PyTorch/torchvision**: If using CUDA, you may need to install a different PyTorch wheel for your CUDA version. Refer to [PyTorch Get Started](https://pytorch.org/get-started/locally/) for the correct installation command.
---

### Environment Setup

1. **Create a virtual environment**

   ```bash
   pyenv virtualenv 3.10.8 fisheye-dev
   ```

2. **Configure Python version with pyenv**

   ```bash
   echo fisheye-dev > .python-version
   ```

3. **Verify the environment is available**

   ```bash
   pyenv versions
   ```

4. **Activate the virtual environment (if needed)**

   ```bash
   pyenv activate fisheye-dev
   ```

5. **Configure Poetry**

   ```bash
   poetry config virtualenvs.create false
   poetry config virtualenvs.in-project true
   poetry config keyring.enabled false
   poetry env use $(which python)
   ```

6. **Install dependencies**

   ```bash
   poetry install
   ```

7. **Activate the environment**

   ```bash
   poetry shell
   ```

   💡 Newer versions of Poetry use:

   ```bash
   poetry env activate
   ```

---

### Pre-commit Hooks

Enable formatting and linting hooks:

```bash
pre-commit install
```

---

### Make the Runner Executable

```bash
chmod +x run.sh
```

---

## Running FishEye

### 1. Select Your Platform Configuration

Open `configs/config.yaml` and locate the `defaults` section:

```yaml
defaults:
  - platform: cpu  # Update with your platform-specific config
  - override hydra/hydra_logging: none
  - override hydra/job_logging: none
```

* The `platform` value **must match a YAML file** in `configs/platform/`.
* You must choose the configuration that corresponds to your **operating system and hardware** (e.g., `cpu`, `cuda`).

💡 Platform-specific settings (e.g., CUDA vs CPU) are defined in `configs/platform/` and may be modified as needed.

Each platform config also defines the model setup:

```yaml
model:
  type: yolov5
  weights: weights/cfc_detect_yolov5s_v1.pt
  device: cuda:0
```

* Model weight paths are relative to the project root.
* Pretrained weights are **downloaded automatically on first run** from GitHub releases. An internet connection is required the first time; subsequent runs use the cached file.
* If automatic download fails, weights can be placed manually at the configured path. Release assets are available at:
  [https://github.com/fisheye-sonar/fisheye/releases](https://github.com/fisheye-sonar/fisheye/releases)

---

### 2. Configure Input and Output Paths

In `configs/config.yaml`, set:

```yaml
input_path:
output_dir:
```

* **`input_path`** may point to:

  * A directory containing multiple `.aris` files, or
  * A single `.aris` file

* **`output_dir`** (optional):

  * If omitted, outputs are written alongside the input files
  * If provided, all outputs are written to this directory

Example:

```yaml
input_path: /home/fisheye/data/
output_dir: /home/fisheye/outputs/
```

---

### 3. (Optional) Configure Export Outputs

Control which outputs are generated using `export_options`:

```yaml
export_options: ["summary_csv", "detailed_csv", "fc", "xml"]
```

Available options:

* `summary_csv` – aggregated summary of counts
* `detailed_csv` – per-file count details
* `fc` – fish count output
* `xml` – MarkedFishMeasurements XML
* `mot` – MOT-style tracking output

Defaults: `summary_csv`, `detailed_csv`, `fc`, `xml`

---

### 4. (Optional) Configure Upstream Direction

When viewing sonar imagery, fish may appear to swim left-to-right or right-to-left. **The image itself does not encode which direction is upstream**—this must be provided as external knowledge.

Set the upstream direction in `configs/config.yaml`:

```yaml
upstream_direction: left  # or right
```

---

### 5. (Optional) Configure Distance Offset

FishEye places markers directly on detected fish by default. You may optionally apply a distance offset (in meters):

```yaml
distance_offset: 0.0
```

* Positive values move markers farther from the sonar
* Negative values move markers closer

---

### 6. Run the Pipeline

From the project root:

```bash
cd ~/home/fisheye/code/fisheye/
./run.sh
```

---

## Saving Echograms for Other Methods

Echogram loading is for exporting sonar data to other methods. YOLO continues to
operate on image frames; this feature does not add echogram detection.

Save a three-channel echogram without allocating warped frame images:

```python
import numpy as np

from fisheye.dataloaders.didson.pyDIDSON import DIDSON
from fisheye.enums import EchogramChannel

sonar = DIDSON("recording.aris")  # .ddf is also supported
channels = [EchogramChannel.BGS, EchogramChannel.BGS_ANGLE, EchogramChannel.RAW]
echogram = sonar.load_echogram(echogram_channels=channels, chunk_size=256)
np.save("recording_echogram.npy", echogram)
```

The direct loader returns a float32 array shaped `(frames, range_samples, channels)`.
It processes at most `chunk_size` beam frames at once, after estimating the
background from at most `num_frames_bg_subtract` leading frames. The full echogram
array remains in memory. `start_frame` is inclusive and `end_frame` is exclusive;
`end_frame=-1` selects the end of the recording. The last selected frame is omitted
for alignment with the image pipeline's temporal differences. Empty and one-frame
ranges return an empty array; invalid ranges raise `ValueError`.

For batched export, use the standard loader with explicit independent outputs:

```python
from fisheye.configs import BaseDatasetConfig
from fisheye.dataloaders import create_dataloader

loader, dataset = create_dataloader(BaseDatasetConfig(
    filepath="recording.aris",
    return_frames=False,
    return_unwarped=False,
    return_echogram=True,
    echogram_channels=channels,
))
for index, (_, _, _, batch_echogram, _) in enumerate(loader):
    np.save(f"echogram_{index:05d}.npy", batch_echogram.numpy())
```

Channel order follows `echogram_channels`, which accepts enums or their string
values. An optional trailing `None` is ignored; `"0"` adds a zero-filled channel.
The default remains `["bgs", "bgs_angle"]`. Request `["raw", "angle"]` for raw
intensity with its angle, or append `"raw"` to the default for a third channel.

| Channel | Values |
| --- | --- |
| `raw` | Maximum beam intensity divided by 255, in `[0, 1]` |
| `bgs` | Maximum background-subtracted intensity divided by the background normalization scale; signed and not clipped |
| `angle` | Raw intensity's peak beam index divided by beam count, minus 0.5 |
| `bgs_angle` | Background-subtracted intensity's peak beam index divided by beam count, minus 0.5 |
| `center_line` | Intensity at the middle beam divided by 255, in `[0, 1]` |
| `0` | Zero |

`do_bg_subtract` controls frame preprocessing. Echogram background subtraction
is determined independently by the selected channels. Explicit echogram-only
loading uses the requested background frame count regardless of batch size.

Existing calls that omit `return_frames` keep the original output contract:
`return_unwarped=True` selects the primary image representation and the auxiliary
unwarped output stays raw. The old `return_echogram_with_bg_subtracted=False`
argument remains supported, including its original raw intensity scale `[0, 255]`.
Explicit `echogram_channels` takes precedence over that argument and uses the
channel scales above.

Setting `return_frames=True` or `False` opts into independent outputs: the primary
slot contains warped frames or `None`, the auxiliary unwarped slot is controlled by
`return_unwarped`, and the echogram slot by `return_echogram`. With BGS enabled,
returned images have three channels (original, BGS, temporal difference); without
BGS, warped images have one channel and auxiliary unwarped frames stay raw.
`YOLODatasetConfig(return_frames=False)` is rejected; use `BaseDatasetConfig` for
saving echograms.

---

## Building Training Datasets (Beta)

⚠️ This feature is under active development.

The dataset builder creates image–annotation pairs from **ARIS/DIDSON recordings** and **ARISFish XML exports**.

### Inputs

`DatasetBuilder` expects:

* `aris_dir`: directory containing `.aris` files
* `xml_dir`: directory containing ARISFish XML files

  * Expected naming: `FCe_<ARIS_STEM>_ID_.xml`
* `out_dir`: output directory for images and annotations
* Optional:

  * `dataset_format` (default: YOLO)
  * `padding`
  * `min_padding_px`

---

### Running the Builder

**Option 1: Configure via YAML**

Edit `configs/dataset.yaml`, then run:

```bash
python build_dataset.py
```

**Option 2: Override via CLI**

```bash
python build_dataset.py \
  aris_dir=/path/to/aris \
  xml_dir=/path/to/xml \
  out_dir=/path/to/output
```

---

### Outputs

The builder produces:

* `out_dir/images/` – extracted JPEG frames
* `out_dir/<format>/` – annotations in the chosen dataset format

  * Example: `out_dir/yolo/`

---

## Did you find a bug?
If you're experiencing a bug or unexpected behavior in the FishEye software, please [submit an issue in GitHub](https://github.com/fisheye-sonar/fisheye/issues/new?template=bug_report.md). This helps us identify and fix the issue quickly.

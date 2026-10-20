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

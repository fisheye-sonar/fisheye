# FishEye 
FishEye is a Python library for working with imaging sonar data, designed to automate processing with computer vision algorithms. It provides data conversion routines, PyTorch Dataset and DataLoader implementations, inference code, and pretrained models for common computer vision tasks such as fish detection in imaging sonar.


FishEye simplifies the process of loading, preprocessing, and analyzing sonar data, making it easier to integrate into machine learning workflows.

## Installation and Setup
### Step 0: Software Requirements
- Python 3.10.14  (Recommendation: use [pyenv](https://github.com/pyenv/pyenv) for python and virtual environment management)
- [Poetry](https://python-poetry.org/docs/) (dependency management & packaging)

### Step 1: Environment Setup
1. Create virtual environment. 

`pyenv virtualenv 3.10.8 fisheye-dev`

2. Copy virtual environment name so poetry can access and automatically enable.

`echo fisheye-dev > .python-version`

3. Check pyenv virtual environment is activated

`pyenv versions`

4. Activate pyenv virtualenv manually 

`pyenv activate fisheye-dev`

5. Update the following poetry configs:

`poetry config virtualenvs.create false`

`poetry config virtualenvs.in-project true`

`poetry config keyring.enabled false`

`poetry env use $(which python)`


6. Install project's dependencies

`poetry install`

7. Activate the virtual environment

`poetry shell`

💡 If using a new version of poetry, command has changed to `poetry env activate`

### Step 2: Activate pre-commit hooks
Run `pre-commit install`.

### Step 3: Make run.sh executable
Run `chmod +x run.sh`


## How to Run the App

1. **Open the main config file**  
   Open `configs/config.yaml` in any text editor (like VS Code, Notepad, or TextEdit) and locate the following lines:

   ```
   input_path:
   output_dir:

2. **Update device/platform in the config**

   In configs/config.yaml, find the platform setting under defaults:

   ```defaults:
     - platform: cpu  # Update with your platform or optimized config
     - override hydra/hydra_logging: none
     - override hydra/job_logging: none
   ```
    By default, it is set to cpu. The value of platform must match the name of a YAML file in configs/platform/. For example, to use CUDA:
    `- platform: cuda`.

    💡 Other platform-specific configs are located in configs/platform/. You can switch them or modify them as needed.

    Each platform YAML (e.g. `cuda.yaml`), also defines the model settings:
    ```
   model:
      type: yolov5
      weights: weights/cfc_detect_yolov5s_v0.pt
      device: cuda:0
   ```
   The weights path is relative to the root of the fisheye project. The model file should live at the project root or in a subdirectory like weights/. Model weights are provided as [downloadable assets in GitHub releases](https://github.com/fisheye-sonar/fisheye/releases). If a new version is released, download the updated weights and place them in the specified path.

3. **Update input and output paths**
   - `input_path` can point to either:
     - A directory containing multiple ARIS files
     - A single ARIS file
   - `output_dir` (optional):
     - If not specified, output files will be written to the same directory as the ARIS file(s).
     - If specified, all output files are written directly to that location

   Examples:
    ```
   input_path: /home/fisheye/
   output_dir:
   ```
    Or:
    ```
   input_path: /home/fisheye/
   output_dir: /home/fisheye/outputs/
   ```
4. **(Optional) Configure distance offset**

    By default, FishEye places markers directly on detected fish in ARISFish. You can configure an optional `distance_offset` value (in meters) to adjust marker placement slightly above or below the fish. In `configs/config.yaml`, modify the following field:
   ```
   distance_offset: 0.0  # Offset in meters - can be integer or float 
   ```
   
    If not specified, the default is 0.0 (no offset). 
   - A positive value (e.g., 1.0) moves the marker 1 meter farther from the sonar camera. 
   - A negative value (e.g., -1.0) places the marker 1 meter closer, effectively positioning it below the fish in the image.

5. **Run the script**

   Once the values are set, open your Terminal. Navigate to the project folder using the cd command. 

       `cd ~/home/fisheye/code/fisheye/`

    Now run the script:

       `./run.sh`

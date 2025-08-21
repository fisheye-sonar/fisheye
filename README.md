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

### Step 4: Make run.sh executable
Run `chmod +x run.sh`


## How to Run the App
1. Open the config file located at `configs/config.yaml` in any text editor (like VS Code, Notepad, or TextEdit), and look for lines like this:

```
    input_path:
    output_dir:  
```    

2. Update to your path locations
   - `input_path` can point to either:
     - A directory contianing multiple ARIS files
     - A single ARIS file
   - `output_dir` (optional):
     - If not specified, output files will be written to the same directory where the ARIS file(s) are located
     - If specified, all output files are written directly to that location

   Example:
    ```
    input_path: /home/fisheye/
    output_dir:
   ```
    Or
    ```
    input_path: /home/fisheye/
    output_dir: /home/fisheye/outputs/
    ```
        

3. Once the values are set, open your Terminal. Navigate to the project folder using the cd command. 

    `cd ~/home/fisheye/code/fisheye/`

    Now run the script:

    `./run.sh`

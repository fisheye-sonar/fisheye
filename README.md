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

5. Make sure virtualenv creation is disabled:

`poetry config virtualenvs.create false`

6. Make sure to enable using virtualenv in project 

`poetry config virtualenvs.in-project true`

7. Tell Poetry to use the current Python executable inside this activated env

`poetry env use $(which python)`

8. Install project's dependencies. If on Mac, run: 

   `poetry install`

   If running on Linux or Windows, run:
   `poetry install --without dev --without test --without mac`

9. Activate the virtual environment

   `poetry shell`

💡 If using a new version of poetry, command has changed to `poetry env activate`

### Step 2: Activate pre-commit hooks
Run `pre-commit install`.

### Step 4: Make run.sh executable
Run `chmod +x run.sh`


## How to Run the App
1. Open the run.sh file in any text editor (like VS Code, Notepad, or TextEdit), and look for lines like this:

    `DATA_DIR=${1:-"./data"}`
    
    `MODEL_PATH=${2:-"./models/model.pt"}`    

    `EXPORT=${3:-"summary_csv,detailed_csv,txt"}`
 
    `OUTPUT_DIR=${4:-"./output"}`

You can change the default values (inside the "") to whatever you want:
    
    `DATA_DIR=${1:-"/home/fisheye/my-folder"}`

💡 If you’re unsure how to edit .sh files, just right-click the file and open it with any text editor.

2. Once the values are set, open your Terminal. Navigate to the project folder using the cd command. 

    `cd ~/home/fisheye/code/fisheye/`

    Now run the script:

    `./run.sh`

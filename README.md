# FishEye 
FishEye is a Python library for working with imaging sonar data, designed to automate processing with computer vision algorithms. It provides data conversion routines, PyTorch Dataset and DataLoader implementations, inference code, and pretrained models for common computer vision tasks such as fish detection in imaging sonar.


FishEye simplifies the process of loading, preprocessing, and analyzing sonar data, making it easier to integrate into machine learning workflows.


## Step 0: Software Requirements
- Python 3.10.14  (Recommendation: use [pyenv](https://github.com/pyenv/pyenv) for python and virtual environment management)
- [Poetry](https://python-poetry.org/docs/) (dependency management & packaging)

## Step 1: Environment Setup
1. Create virtual environment. 

`pyenv virtualenv 3.10.8 fisheye-dev`

2. Copy virutal environment name so poetry can access and automatically enable.

`echo fisheye-dev > .python-version`

3. Install project's dependencies

`poetry install`

## Step 2: Activate pre-commit hooks
Run `pre-commit install`.
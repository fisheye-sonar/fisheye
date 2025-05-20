#!/bin/bash

### MAKE CHANGES HERE ###

# Path to the folder containing your ARIS or DIDSON files
DATA_DIR=${1:-"./data"}

# Path to your trained model weights file
MODEL_PATH=${2:-"./models/model.pt"}

# What to export. Options include: summary_csv, detailed_csv, txt, mot, none. Use "none" to skip exporting.
EXPORT=${3:-"summary_csv,detailed_csv,txt"}

# Where to save the output files.
# You can choose any folder. If you save to the same folder as your ARIS/DIDSON files, ARISFish Software will be able to
# read the model outputs and display results in its interface. Otherwise, you'll need to manually copy the .txt output
# files to the same folder later if you want to view them in the ARISFish interface.
OUTPUT_DIR=${4:-"./data/"}

### END OF CHANGES ###


### DO NOT CHANGE ANYTHING BELOW THIS LINE ###
poetry run python fisheye/main.py \
  --path "$DATA_DIR" \
  --weights "$MODEL_PATH" \
  --export_options "$EXPORT" \
  --output_dir "$OUTPUT_DIR"

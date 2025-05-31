#!/bin/bash

### MAKE CHANGES HERE ###

# Path to the folder containing your ARIS or DIDSON files
INPUT_DIR=${1:-""}

# What to export. Options include: summary_csv, detailed_csv, txt, mot, none. Use "none" to skip exporting.
EXPORT_OPTIONS=${3:-"summary_csv,detailed_csv,txt"}

# Where to save the output files.
# You can choose any folder. If you save to the same folder as your ARIS/DIDSON files, ARISFish Software will be able to
# read the model outputs and display results in its interface. Otherwise, you'll need to manually copy the .txt output
# files to the same location of the ARIS file if you want to view them in the ARISFish interface.
# If empty or not provided, defaults to same as INPUT_DIR
RAW_OUTPUT_DIR=${4:-""}
OUTPUT_DIR=${RAW_OUTPUT_DIR:-$INPUT_DIR}

### END OF CHANGES ###


### DO NOT CHANGE ANYTHING BELOW THIS LINE ###
poetry run python fisheye/main.py \
  --path "$INPUT_DIR" \
  --export_options "$EXPORT_OPTIONS" \
  --output_dir "$OUTPUT_DIR"

#!/bin/bash

### MAKE CHANGES HERE ###

# Path to your ARIS or DIDSON data
# You can provide either:
# - a folder containing one or more ARIS/DIDSON files, or
# - a single ARIS/DIDSON file
# INPUT_DIR=${1:-"/home/mahobley/Data/CFC22/aris_files/nushagak/RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000.aris"}
INPUT_DIR=${1:-"/home/mahobley/Data/CFC22/aris_files/"}
# INPUT_DIR=${1:-"/home/mahobley/Data/CFC22/aris_files/kenai-train/2018-05-26-JD146_LeftFar_Stratum1_Set1_LO_2018-05-26_080004.aris"}

# What to export. Options include: summary_csv, detailed_csv, txt, mot, none. Use "none" to skip exporting.
EXPORT_OPTIONS=${3:-"summary_csv,detailed_csv,txt"}

# Where to save the output files.
# You can choose any folder. If you save to the same folder as your ARIS/DIDSON files, ARISFish Software will be able to
# read the model outputs and display results in its interface. Otherwise, you'll need to manually copy the .txt output
# files to the same location of the ARIS file if you want to view them in the ARISFish interface.
# If empty or not provided, defaults to same as INPUT_DIR
RESULTS_FOLDER=${4:-"/home/mahobley/Code/fisheye/results"}
OUTPUT_DIR=${RESULTS_FOLDER:-$INPUT_DIR}

# Map input directory structure to output directory structure.
MAP_INPUT_DIR_STRUCTURE_TO_OUTPUT="--map_input_dir_structure_to_output"
# Put all results in the same folder.
# MAP_INPUT_DIR_STRUCTURE_TO_OUTPUT="--no-map_input_dir_structure_to_output"

### END OF CHANGES ###


### DO NOT CHANGE ANYTHING BELOW THIS LINE ###
poetry run python fisheye/main.py \
  --path "$INPUT_DIR" \
  --export_options "$EXPORT_OPTIONS" \
  --output_dir "$OUTPUT_DIR" \
  $MAP_INPUT_DIR_STRUCTURE_TO_OUTPUT

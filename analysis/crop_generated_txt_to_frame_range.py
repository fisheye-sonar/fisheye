import os
import sys
import argparse
import re
from pathlib import Path


def parse_frame_id(line):
    """
    Extract frame ID from a line. Assumes frame ID is a number in the line.
    Returns None if no valid frame ID is found.
    """
    # Look for numbers in the line - adjust this pattern based on your actual data format
    numbers = re.findall(r"\d+", line)
    if numbers:
        # Assuming the first number is the frame ID - adjust as needed
        return int(numbers[2])
    return None


def crop_txt_to_frame_range(
    input_file, start_frame, end_frame, output_file=None, verbose=False
):
    """
    Filter a file by frame ID range and save to new file.

    Args:
        input_file (str): Path to input file
        start_frame (int): Minimum frame ID to include
        end_frame (int): Maximum frame ID to include
        output_file (str, optional): Output file path. If None, auto-generates name.

    Returns:
        str: Path to output file
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Generate output filename if not provided
    if output_file is None:
        base_name = input_path.stem
        suffix = input_path.suffix
        output_file = f"{base_name.rstrip('_')}_{start_frame}_{end_frame}{suffix}"
        output_path = input_path.parent / output_file
    else:
        output_path = Path(output_file)

    filtered_lines = []
    total_lines = 0
    kept_lines = 0
    met_a_result_line = False

    if verbose:
        print(f"Processing file: {input_file}")
        print(f"Frame range: {start_frame} to {end_frame}")

    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1

            # Skip empty lines
            # if not line.strip():
            #     continue

            if (
                not line.strip()
                or line.startswith("***")
                or line.startswith("---")
                or line.startswith("File")
                or line == "\n"
            ):
                continue

            frame_id = parse_frame_id(line)

            if frame_id is None:
                print(
                    f"Warning: Could not parse frame ID from line {line_num}: {line.strip()}"
                )
                if not met_a_result_line:
                    filtered_lines.append(line)
                continue

            # Check if frame ID is within range
            if start_frame <= frame_id <= end_frame:
                met_a_result_line = True
                filtered_lines.append(line)
                kept_lines += 1

    # Write filtered data to output file
    with open(output_path, "w") as f:
        f.writelines(filtered_lines)
    if verbose:
        print(f"Total lines processed: {total_lines}")
        print(f"Lines kept: {kept_lines}")
        print(f"Lines filtered out: {total_lines - kept_lines}")
    print(f"Output saved to: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    start_frame = 285
    end_frame = 885
    input_dir = "/home/mahobley/Code/fisheye/results/"
    output_dir = "/home/mahobley/Code/fisheye/results/"
    input_fn = "FCe_2018-05-26-JD146_LeftFar_Stratum1_Set1_LO_2018-05-26_080004_ID_.txt"
    input_fn = "FCe_RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_ID_.txt"
    output_fn = f"{os.path.splitext(input_fn)[0]}_{start_frame}_{end_frame}_cropped.txt"
    input_path = os.path.join(input_dir, input_fn)
    output_path = os.path.join(output_dir, output_fn)
    crop_txt_to_frame_range(input_path, start_frame, end_frame, output_path)

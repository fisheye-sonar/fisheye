import argparse
import os
import glob
import shutil


def flatten_files(source_dir, target_dir, target_ext="*.txt"):
    """
    Moves all .txt files from subdirectories in `source_dir` into `target_dir`, flattening the structure.

    Args:
        source_dir (str): The root directory containing subdirectories with .txt files.
        target_dir (str): The directory where all .txt files will be moved.
        target_ext (str): The file extension used to move all X files.
    """
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Find all .txt files recursively in the source directory
    files = glob.glob(os.path.join(source_dir, "**", target_ext), recursive=True)

    for file in files:
        filename = os.path.basename(file)

        # Construct destination path
        dest_path = os.path.join(target_dir, filename)

        print(f"Moving {filename} to {dest_path}")

        # Avoid overwriting files with the same name
        if os.path.exists(dest_path):
            print(f"Inside if statement")
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                new_filename = f"{base}_{counter}{ext}"
                print(new_filename)
                break
                dest_path = os.path.join(target_dir, new_filename)
                counter += 1

        # Move file to target directory
        shutil.move(file, dest_path)

    print(f"Moved {len(files)} files to {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir", type=str, help="ARIS/DIDSON filepath", required=True
    )
    parser.add_argument(
        "--target_dir", type=str, help="Path to MOT txt file(s)", required=True
    )
    parser.add_argument(
        "--target_ext",
        type=str,
        help="Target extension e.g. *.txt or *.aris",
        required=False,
    )
    args = parser.parse_args()

    flatten_files(args.source_dir, args.target_dir, args.target_ext)

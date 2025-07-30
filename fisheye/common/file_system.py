import os
from pathlib import Path
from typing import List, Union

from fisheye.enums import ValidExtensions, IGNORED_DIR_NAMES, IGNORED_FILE_PREFIXES


def is_valid_file(file_path: Path) -> bool:
    """Check if the file is valid based on the extension."""
    return file_path.is_file() and file_path.suffix in {
        e.value for e in ValidExtensions
    }


def is_valid_dir(dir_path: Path) -> bool:
    """Check if the directory is valid."""
    return dir_path.is_dir()


def get_all_valid_files_in_dir(path: Path) -> List[Path]:
    """Get all valid files in a directory."""
    valid_files = []
    for root, dirs, files in os.walk(path):
        # Skip ignored system directories
        dirs[:] = [
            d for d in dirs if d not in IGNORED_DIR_NAMES and not d.startswith(".")
        ]

        for file in files:
            # Skip files with ignored prefixes
            if file.startswith(".") or any(
                file.startswith(prefix) for prefix in IGNORED_FILE_PREFIXES
            ):
                continue

            file_path = Path(root) / file
            if is_valid_file(file_path):
                valid_files.append(file_path)

    return valid_files


def get_valid_files(inputs: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
    """Return all valid files from one or more input paths (file or directory)."""
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    results = []
    for input_path in inputs:
        path = Path(input_path)

        if is_valid_file(path):
            results.append(path)

        elif is_valid_dir(path):
            results.extend(get_all_valid_files_in_dir(path))

    return results

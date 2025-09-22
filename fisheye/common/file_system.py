import os
from pathlib import Path
from typing import List, Union

import structlog

from fisheye.enums import ValidExtensions, IGNORED_DIR_NAMES, IGNORED_FILE_PREFIXES

logger = structlog.getLogger(__name__)


def is_valid_file(file_path: Path) -> bool:
    """Check if the file is valid based on the extension."""
    return file_path.is_file() and file_path.suffix in {
        e.value for e in ValidExtensions
    }


def is_valid_dir(dir_path: Union[Path, str]) -> bool:
    """Check if the directory is valid."""
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

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


def get_valid_files(
    inputs: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path],
    rerun_and_overwrite_already_processed_files: bool = False,
) -> List[Path]:
    """Return all valid files from one or more input paths (file or directory)."""
    missing_output_dir = not output_dir
    if output_dir:
        output_dir = Path(output_dir)

    if isinstance(inputs, (str, Path)):
        inputs = [inputs]

    results = []
    for input_path in inputs:
        path = Path(input_path)

        if is_valid_file(path):
            candidate_files = [path]

        elif is_valid_dir(path):
            candidate_files = get_all_valid_files_in_dir(path)

        else:
            continue

        # Exclude files if they have already been processed
        for file in candidate_files:
            if missing_output_dir:
                output_dir = file.parent
            prefix = f"FCe_{file.stem}_ID_"
            expected_output_file_name = f"{prefix}.txt"
            existing_files = [f for f in os.listdir(output_dir) if f.startswith(prefix)]
            if not (output_dir / expected_output_file_name).exists():
                results.append(file)
            else:
                logger.warning("FC_txt_exists_already", file=file.name)
                if not rerun_and_overwrite_already_processed_files:
                    continue

                logger.warning(
                    "overwrite_existing_fcs set to True so will run on this file anyway",
                    file=file.name,
                )

                if len(existing_files) == 1:
                    # change file to have a 0 at the end
                    new_num = 0
                else:
                    # change the file to have the next number
                    existing_nums = [
                        int(f.removeprefix(prefix).split(".")[0])
                        for f in existing_files
                        if f.removeprefix(prefix).split(".")[0].isdigit()
                    ]
                    new_num = max(existing_nums) + 1
                new_file_name = f"{prefix}{new_num}.txt"
                logger.warning(
                    f"moving the old file to a {expected_output_file_name}->{new_file_name}"
                )
                os.rename(
                    output_dir / expected_output_file_name,
                    output_dir / new_file_name,
                )

                results.append(file)

    return results

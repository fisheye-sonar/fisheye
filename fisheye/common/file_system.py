import os
from pathlib import Path
from typing import List, Union

import structlog

from fisheye.enums import ValidExtensions, IGNORED_DIR_NAMES, IGNORED_FILE_PREFIXES

logger = structlog.getLogger(__name__)


def _should_ignore_path(fp: Path) -> bool:
    """Return True if the given path should be ignored based on common rules."""
    parts = fp.parts

    # Skip hidden files or files in hidden directories
    if any(part.startswith(".") for part in parts):
        return True

    # Skip paths that go through ignored directories
    if any(part in IGNORED_DIR_NAMES for part in parts):
        return True

    # Skip files whose *basename* starts with an ignored prefix
    if any(fp.name.startswith(prefix) for prefix in IGNORED_FILE_PREFIXES):
        return True

    # Skip temp/backup files
    if fp.name.startswith("~"):
        return True

    return False


def find_aris_xml_files(base_path: Path, pattern: str) -> List[Path]:
    """Find all ARIS XML files in a directory."""
    return [
        fp
        for fp in base_path.rglob(pattern)
        if fp.is_file() and not _should_ignore_path(fp)
    ]


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
        dirs[:] = [d for d in dirs if not _should_ignore_path(Path(root) / d)]

        for file in files:
            file_path = Path(root) / file

            if _should_ignore_path(file_path):
                continue

            if is_valid_file(file_path):
                valid_files.append(file_path)

    return valid_files


def get_valid_files(
    inputs: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path],
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
            expected_output_file = output_dir / f"FCe_{file.stem}_ID_.txt"
            if expected_output_file.exists():
                logger.warning("FC_txt_exists_already", file=file.name)
                continue

            results.append(file)

    return results

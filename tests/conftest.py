import os
from pathlib import Path

import pytest

ARIS_FILE = os.path.join(os.path.dirname(__file__), "data", "sample.aris")
DDF_FILE = os.path.join(os.path.dirname(__file__), "data", "sample.ddf")
CORRUPTED_FILE = os.path.join(os.path.dirname(__file__), "data", "corrupted.aris")
SHORTENED_DDF_FILE = os.path.join(
    os.path.dirname(__file__), "data", "shortened_clip.ddf"
)
INVALID_FRAME_INDICES = os.path.join(
    os.path.dirname(__file__), "data", "invalid_frame_indices.aris"
)


@pytest.fixture(scope="session")
def beam_widths_path():
    """Fixture to get the path to the beam_widths directory."""
    project_root = Path(__file__).parent.parent  # Get the root of the fisheye/ project
    beam_widths_dir = (
        project_root / "fisheye" / "beam_widths"
    )  # Navigate to beam_widths
    return beam_widths_dir

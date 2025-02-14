import os
from pathlib import Path

import pytest

ARIS_FILE = os.path.join(os.path.dirname(__file__), "sample.aris")
DDF_FILE = os.path.join(os.path.dirname(__file__), "sample.ddf")
CORRUPTED_FILE = os.path.join(os.path.dirname(__file__), "corrupted.aris")
SHORTENED_DDF_FILE = os.path.join(os.path.dirname(__file__), "shortened_clip.ddf")


@pytest.fixture(scope="session")
def beam_widths_path():
    """Fixture to get the path to the beam_widths directory."""
    project_root = Path(__file__).parent.parent  # Get the root of the fisheye/ project
    beam_widths_dir = (
        project_root / "fisheye" / "beam_widths"
    )  # Navigate to beam_widths
    return beam_widths_dir

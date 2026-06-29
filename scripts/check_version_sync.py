"""Pre-commit check: _APP_VERSION in version.py must match pyproject.toml."""

import sys
import tomli
from pathlib import Path

root = Path(__file__).parent.parent

with open(root / "pyproject.toml", "rb") as f:
    pyproject_version = tomli.load(f)["tool"]["poetry"]["version"].lstrip("v")

from fisheye.version import _APP_VERSION

if _APP_VERSION != pyproject_version:
    print(
        f"ERROR: Version mismatch — "
        f"fisheye/version.py _APP_VERSION={_APP_VERSION!r} "
        f"but pyproject.toml version={pyproject_version!r}"
    )
    sys.exit(1)

print(f"OK: versions in sync ({_APP_VERSION})")

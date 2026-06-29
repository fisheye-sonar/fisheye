"""Pre-commit check: _APP_VERSION in version.py must match pyproject.toml."""

import re
import sys
import tomli
from pathlib import Path

root = Path(__file__).parent.parent

with open(root / "pyproject.toml", "rb") as f:
    pyproject_version = tomli.load(f)["tool"]["poetry"]["version"].lstrip("v")

version_py = (root / "fisheye" / "version.py").read_text()
match = re.search(r'^_APP_VERSION\s*=\s*["\']([^"\']+)["\']', version_py, re.MULTILINE)

if not match:
    print("ERROR: Could not find _APP_VERSION in fisheye/version.py")
    sys.exit(1)

code_version = match.group(1)

if code_version != pyproject_version:
    print(
        f"ERROR: Version mismatch — "
        f"fisheye/version.py _APP_VERSION={code_version!r} "
        f"but pyproject.toml version={pyproject_version!r}"
    )
    sys.exit(1)

print(f"OK: versions in sync ({code_version})")

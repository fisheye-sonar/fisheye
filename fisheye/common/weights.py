import json
import urllib.error
import urllib.request
from pathlib import Path

import structlog

logger = structlog.get_logger()

_GITHUB_REPO = "fisheye-sonar/fisheye"


def _find_release_url(filename: str) -> str:
    """Find download URL for a release asset including pre-releases."""
    api_url = f"https://api.github.com/repos/{_GITHUB_REPO}/releases"
    req = urllib.request.Request(
        api_url, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req) as resp:
        releases = json.loads(resp.read())

    for release in releases:
        for asset in release.get("assets", []):
            if asset["name"] == filename:
                return asset["browser_download_url"]

    raise FileNotFoundError(
        f"Could not find '{filename}' in any release of {_GITHUB_REPO}. "
        f"Place the file manually at the configured weights path."
    )


def ensure_weights(local_path: str) -> None:
    """Download detector weights from GitHub releases if not present locally."""
    path = Path(local_path)
    if path.exists():
        return

    filename = path.name
    url = _find_release_url(filename)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    logger.info("downloading_weights", weights_file=filename, url=url)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
            while chunk := response.read(1 << 16):
                f.write(chunk)

        tmp_path.rename(path)
        logger.info("weights_downloaded", weights_file=filename)

    except urllib.error.HTTPError as e:
        tmp_path.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"Could not download weights '{filename}' (HTTP {e.code}). "
            f"Place the file manually at: {path}"
        ) from e
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

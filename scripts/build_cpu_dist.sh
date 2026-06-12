#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Installing GUI build dependencies..."
poetry install --with gui

echo "==> Verifying torch is CPU build..."
poetry run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    raise SystemExit(
        "CUDA torch is still installed. Build the desktop bundle from a CPU-only environment."
    )
PY

echo "==> Running PyInstaller..."
poetry run pyinstaller --noconfirm --clean fisheye_app.spec

echo "==> Done. Output: dist/FisheyeArisSalmonDetection/"
echo "    Copy weights/*.pt to dist/FisheyeArisSalmonDetection/weights/ before distributing."


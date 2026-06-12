$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> Installing GUI build dependencies..."
poetry install --with gui

Write-Host "==> Verifying torch is CPU build..."
poetry run python -c @"
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    raise SystemExit(
        'CUDA torch is still installed. Build the desktop bundle from a CPU-only environment.'
    )
"@

Write-Host "==> Running PyInstaller..."
poetry run pyinstaller --noconfirm --clean fisheye_app.spec

Write-Host "==> Done. Output: dist\FisheyeArisSalmonDetection\"
Write-Host "    Copy weights\*.pt to dist\FisheyeArisSalmonDetection\weights\ before distributing."

#!/usr/bin/env bash
set -e

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)
echo "Detected OS: $OS, architecture: $ARCH"

# Paths to local wheels (Jetson)
TORCH_WHEEL="./torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
TORCHVISION_WHEEL="./torchvision-0.15.1-cp38-cp38-linux_aarch64.whl"

if [ "$ARCH" = "aarch64" ]; then
    echo "Jetson detected: installing local wheels..."

    pip install numpy

    if [ -f "$TORCH_WHEEL" ]; then
        pip install --no-deps "$TORCH_WHEEL"
    else
        echo "Torch wheel not found at $TORCH_WHEEL, skipping."
    fi

    if [ -f "$TORCHVISION_WHEEL" ]; then
        pip install --no-deps "$TORCHVISION_WHEEL"
    else
        echo "Torchvision wheel not found at $TORCHVISION_WHEEL, skipping."
    fi

elif [[ "$OS" == "Darwin" ]]; then
    echo "Mac detected: installing torch 2.6.0 from PyPI..."
    pip install "torch==2.6.0"

else
    echo "Other system detected: installing torch and torchvision from PyPI..."
    pip install "torch==2.6.0"
fi

echo "Installing remaining dependencies via Poetry..."
poetry install

echo "Installation complete."


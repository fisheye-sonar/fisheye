from pathlib import Path
from typing import Union
import torch.nn as nn

import os
import torch
import platform
from logging import getLogger

LOGGER = getLogger(__name__)


def get_device_info(device):
    """Returns a formatted string with device information."""
    device_info = []
    for i, d in enumerate(device):
        props = torch.cuda.get_device_properties(int(d))
        device_info.append(
            f"CUDA:{d} ({props.name}, {props.total_memory / (1 << 20):.0f}MiB)"
        )

    return "\n".join(device_info)


def validate_cuda_devices(device):
    """Validate requested CUDA device(s)."""
    if not torch.cuda.is_available():
        raise ValueError(
            f"Invalid CUDA '--device {device}' requested. Use '--device cpu' or specify valid CUDA devices."
        )

    devices = device.split(",")
    if torch.cuda.device_count() < len(devices):
        raise ValueError(
            f"Requested {len(devices)} GPUs, but only {torch.cuda.device_count()} are available."
        )

    return devices


def select_device(device="", batch_size=0):
    """Selects device ('cpu', 'mps', '0', '0,1,2', etc.) for inference."""
    s = f"YOLOv5 🚀 {platform.python_version()} torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")

    if device in ("cpu", "mps"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        arg = "mps" if device == "mps" and torch.backends.mps.is_available() else "cpu"
        s += "MPS\n" if arg == "mps" else "CPU\n"
    else:
        devices = validate_cuda_devices(device) if device else ["0"]
        if batch_size and len(devices) > 1 and batch_size % len(devices) != 0:
            raise ValueError(
                f"Batch size {batch_size} must be a multiple of the number of GPUs {len(devices)}."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        s += get_device_info(devices)
        arg = "cuda:0"

    return torch.device(arg)


class BaseModel(nn.Module):
    """BaseModel

    A base class for detection models to standardize loading weights and running inference.
    """

    def __init__(self, model_path, device, config) -> None:
        """
        Initializes the model class with loading weights and setting the device.

        Args:
            model_path: Path to the model weights.
            device: The device (CPU or GPU) to run the inference on.
        """
        super().__init__()
        self.config = config
        self._setup(model_path, device)

    def __call__(self, *args, **kwargs):
        """Use model instance to make predictions."""
        self.forward(*args, **kwargs)

    def _setup(self, model_path, device):
        if torch.cuda.is_available():
            self.device = select_device("0", batch_size=self.config.batch_size)
        else:
            print("CUDA not available. Using CPU inference.")
            self.device = select_device("cpu", batch_size=self.config.batch_size)

        model = self._load_weights(model_path)
        half = device.type != "cpu"  # half precision only supported on CUDA
        if half:
            model.half()
        self.model.eval()

    def _load_weights(self, model_path):
        """Load model weights from a given file path.

        Args:
            model_path (Union[str, Path]): Path to the model weights.

        Returns:
            nn.Module: The PyTorch model with the loaded weights.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass for the model.

        Args:
            x (torch.Tensor): The input tensor for inference.

        Returns:
            torch.Tensor: The model's output.
        """
        with torch.no_grad():
            return self.model(x, *args, **kwargs)

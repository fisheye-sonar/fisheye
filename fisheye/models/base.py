from pathlib import Path
from typing import Union

import torch


class BaseModel:
    """BaseModel

    A base class for detection models to standardize loading weights and running inference.
    """

    def __init__(self, model_path, device) -> None:
        """
        Initializes the model class with loading weights and setting the device.

        Args:
            model_path: Path to the model weights.
            device: The device (CPU or GPU) to run the inference on.
        """
        super().__init__()
        self._get_model_instance(model_path, device)

    def __call__(self, *args, **kwargs):
        """Use model instance to make predictions."""
        return self.predict(*args, **kwargs)

    def _get_model_instance(self, model_path, device):
        """Sets up the model instance."""
        self.model = self._load_model(model_path, device)
        self.model.eval()

    def _load_model(self, model_path, device):
        """Load model weights from a given file path.

        Args:
            model_path (Union[str, Path]): Path to the model weights.
            device (str): The device (CPU or GPU) to run inference on.

        Returns:
            nn.Module: The PyTorch model with the loaded weights.
        """
        raise NotImplementedError("`_load_model` must be implemented.")

    def predict(self, x, *args, **kwargs) -> torch.Tensor:
        """Forward pass for the model.

        Args:
            x (torch.Tensor): The input tensor for inference.

        Returns:
            torch.Tensor: The model's output.
        """
        with torch.no_grad():
            return self.model(x, *args, **kwargs)

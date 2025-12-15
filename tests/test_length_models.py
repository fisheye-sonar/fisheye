from unittest.mock import patch

import pytest
import torch
from fisheye.lengths.models import UNetHeatmap, HeatmapCNN, get_model


def test_unet_heatmap_forward():
    """Test UNetHeatmap forward pass with different input sizes."""
    model = UNetHeatmap(in_ch=3, out_ch=2)

    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 2, 256, 256)

    # Test with odd size (to check padding/interpolation)
    x = torch.randn(1, 3, 257, 257)
    out = model(x)
    assert out.shape == (1, 2, 257, 257)


def test_heatmap_cnn_forward():
    """Test HeatmapCNN forward pass."""
    model = HeatmapCNN(in_ch=1)

    x = torch.randn(1, 1, 256, 256)
    out = model(x)
    assert out.shape == (1, 2, 256, 256)


def test_get_model_unet():
    """Test get_model factory for UNet."""
    # Create a dummy model to get a valid state dict
    dummy_model = UNetHeatmap(in_ch=3, out_ch=2, use_double_conv=False)
    dummy_state_dict = dummy_model.state_dict()

    with patch("pathlib.Path.exists", return_value=True):
        with patch("torch.load", return_value=dummy_state_dict):
            model = get_model(
                model_type="unet",
                model_input_channels=3,
                unet_double_conv=False,
                weights="/invalid/path/to/weights.pt",  # mocked
                device="cpu",
            )
    assert isinstance(model, UNetHeatmap)


def test_get_model_heatmap_cnn():
    """Test get_model factory for HeatmapCNN."""
    # Create a dummy model to get a valid state dict
    dummy_model = HeatmapCNN(in_ch=1)
    dummy_state_dict = dummy_model.state_dict()

    with patch("pathlib.Path.exists", return_value=True):
        with patch("torch.load", return_value=dummy_state_dict):
            model = get_model(
                model_type="heatmap_cnn",
                model_input_channels=1,
                unet_double_conv=False,
                weights="/path/to/weights.pt",  # Mock path
                device="cpu",
            )
    assert isinstance(model, HeatmapCNN)


def test_get_model_invalid_model_path():
    """Test get_model with invalid weigths path."""
    # Ensure Path.exists() returns False to trigger FileNotFoundError
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            get_model(
                model_type="unet",
                model_input_channels=3,
                unet_double_conv=False,
                weights="/path/to/weights.pt",  # Mock path
                device="cpu",
            )


def test_get_model_invalid_file_type():
    """Test get_model with invalid type."""
    with pytest.raises(ValueError):
        get_model(
            model_type="invalid_type",
            model_input_channels=3,
            unet_double_conv=False,
            weights="/path/to/weights.pt",  # Mock path
            device="cpu",
        )

from unittest.mock import MagicMock, patch

import pytest
import torch

from conftest import ARIS_FILE
from fisheye.configs import (
    ObjectDetectionConfig,
    YOLODatasetConfig,
    YOLOv5ModelConfig,
)
from fisheye.configs.inference import NMSConfig
from fisheye.detect.yolov5 import YOLOv5ObjectDetectionModel
from fisheye.pipelines import ObjectDetectionPipeline


def test_preprocess():
    """Test image preprocessing."""

    dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
    model_cfg = YOLOv5ModelConfig(weights="dummy/path")
    config = ObjectDetectionConfig(model=model_cfg)

    # Mock the model to be used in the pipeline
    mock_model = MagicMock()
    mock_predict_return = torch.rand((1, 6, 6))
    mock_model.predict.return_value = mock_predict_return

    # Patch the _load_model method to return the mocked model
    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):

        pipeline = ObjectDetectionPipeline(config, dataset_cfg)

        mock_image = torch.rand((1, 3, 640, 640))
        preprocessed_image = pipeline.preprocess(mock_image)

        assert preprocessed_image.device.type == model_cfg.device
        assert preprocessed_image.shape == (1, 3, 640, 640)
        assert (preprocessed_image <= 1.0).all()  # Ensure the image is normalized


@pytest.mark.parametrize("use_multithreading", [True, False])
def test_object_detection_pipeline_no_postprocessing(use_multithreading):
    """Test ObjectDetectionPipeline with a mocked `self.model` (patched _load_model)."""

    dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
    batch_size = dataset_cfg.batch_size
    mock_model = MagicMock()

    if use_multithreading:
        # Each call returns shape [1, 30240, 6]
        mock_model.predict.side_effect = [
            torch.rand((1, 30240, 6)) for _ in range(batch_size)
        ]
    else:
        # Single batched output: shape [batch_size, 30240, 6]
        mock_model.predict.return_value = torch.rand((batch_size, 30240, 6))

    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(
            model=model_cfg, use_multithreading=use_multithreading
        )
        pipeline = ObjectDetectionPipeline(config, dataset_cfg)

        # Manually override the dataloader with a mock batch to ensure consistent input shape.
        # This is necessary because when `use_multithreading=True`, the pipeline slices the batch
        # into individual images and runs them in parallel. Mocking ensures a controlled environment for both
        # threading modes.
        img_batch = torch.rand(batch_size, 3, 960, 512)
        shapes = [(960, 512)] * batch_size
        pipeline.dataloader = [(img_batch, None, shapes)]

        pipeline.model = mock_model
        output = pipeline()

        expected_calls = batch_size if use_multithreading else 1
        assert mock_model.predict.call_count == expected_calls

        assert len(output.pred_bboxes) == 1  # One batch
        assert isinstance(output.pred_bboxes[0], torch.Tensor)
        assert output.pred_bboxes[0].shape == (batch_size, 30240, 6)

        assert isinstance(output.image_shape[0], list)
        assert len(output.image_shape[0]) == batch_size
        for img_info in output.image_shape[0]:
            assert isinstance(img_info[0], torch.Size)
            assert img_info[0] == torch.Size([960, 512])

        assert output.width == 512
        assert output.height == 960


@pytest.mark.parametrize(
    "confs,use_multithreading",
    [
        ([0.1], True),
        ([0.1], False),
        ([0.1, 0.3], True),
        ([0.1, 0.3], False),
    ],
)
def test_object_detection_pipeline_w_postprocessing_params(confs, use_multithreading):
    """Test enabling postprocessing parameters with and without multithreading."""

    postprocessing_params = {"nms": [NMSConfig(conf=c) for c in confs]}
    dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
    batch_size = dataset_cfg.batch_size

    mock_model = MagicMock()
    if use_multithreading:
        mock_model.predict.side_effect = [
            torch.rand((1, 30240, 6)) for _ in range(batch_size)
        ]
    else:
        mock_model.predict.return_value = torch.rand((batch_size, 30240, 6))

    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(
            model=model_cfg, use_multithreading=use_multithreading
        )

        pipeline = ObjectDetectionPipeline(config, dataset_cfg, postprocessing_params)

        # Manually override the dataloader with a mock batch to ensure consistent input shape.
        # This is necessary because when `use_multithreading=True`, the pipeline slices the batch
        # into individual images and runs them in parallel. Mocking ensures a controlled environment for both
        # threading modes.
        img_batch = torch.rand(batch_size, 3, 960, 512)
        shapes = [(960, 512)] * batch_size
        pipeline.dataloader = [(img_batch, None, shapes)]

        pipeline.model = mock_model
        output = pipeline()

        expected_calls = batch_size if use_multithreading else 1
        assert mock_model.predict.call_count == expected_calls

        steps = pipeline.postprocessing_steps
        assert len(output) == len(steps)
        assert len(steps) == len(confs)

        for step, conf_val in zip(steps, confs):
            assert callable(step)
            assert step.keywords["nms_config"].conf == conf_val


@pytest.mark.parametrize(
    "postprocessing_param,use_multithreading",
    [
        ({"nms": NMSConfig()}, True),
        ({"nms": NMSConfig()}, False),
        ({"nms": [NMSConfig()]}, True),
        ({"nms": [NMSConfig()]}, False),
        ({"nms": []}, True),
        ({"nms": []}, False),
    ],
)
def test_object_detection_pipeline_diff_postprocessing_structure(
    postprocessing_param, use_multithreading
):
    """Test enabling postprocessing parameters."""

    dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
    batch_size = dataset_cfg.batch_size

    mock_model = MagicMock()
    if use_multithreading:
        mock_model.predict.side_effect = [
            torch.rand((1, 30240, 6)) for _ in range(batch_size)
        ]
    else:
        mock_model.predict.return_value = torch.rand((batch_size, 30240, 6))

    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(
            model=model_cfg, use_multithreading=use_multithreading
        )
        batch_size = dataset_cfg.batch_size

        pipeline = ObjectDetectionPipeline(config, dataset_cfg, postprocessing_param)

        # Manually override the dataloader with a mock batch to ensure consistent input shape.
        # This is necessary because when `use_multithreading=True`, the pipeline slices the batch
        # into individual images and runs them in parallel. Mocking ensures a controlled environment for both
        # threading modes.
        img_batch = torch.rand(batch_size, 3, 960, 512)
        shapes = [(960, 512)] * batch_size
        pipeline.dataloader = [(img_batch, None, shapes)]

        pipeline.model = mock_model
        output = pipeline()

        expected_calls = batch_size if use_multithreading else 1
        assert mock_model.predict.call_count == expected_calls
        steps = pipeline.postprocessing_steps

        if steps:
            assert len(output) == len(steps)


def test_object_detection_pipeline_postprocessing_invalid_params():
    """Test sending invalid postprocessing parameters"""

    # Mock the model to be used in the pipeline
    mock_model = MagicMock()
    mock_predict_return = torch.rand((1, 6, 6))
    mock_model.predict.return_value = mock_predict_return

    # Patch the _load_model method to return the mocked model
    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(model=model_cfg)
        processing_params = {"some_func_name": "some_func"}
        with pytest.raises(
            ValueError, match="Unknown postprocessing step: some_func_name"
        ):
            ObjectDetectionPipeline(config, dataset_cfg, processing_params)

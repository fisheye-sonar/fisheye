from unittest.mock import MagicMock, patch

import pytest
import torch

from conftest import ARIS_FILE
from fisheye.common.file_system import get_all_valid_files_in_dir
from fisheye.common.generic import safe_execution
from fisheye.configs import ObjectDetectionConfig, YOLOv5ModelConfig, YOLODatasetConfig
from fisheye.configs.inference import NMSConfig
from fisheye.configs.inference import TrackerOutput
from fisheye.detect.yolov5 import YOLOv5ObjectDetectionModel
from fisheye.pipelines import ObjectDetectionPipeline
from fisheye.pipelines.pipeline import DetectTrackCountPipeline


def test_preprocess():
    """Test image preprocessing."""

    dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
    model_cfg = YOLOv5ModelConfig(weights="dummy/path")
    config = ObjectDetectionConfig(model=model_cfg)

    # Mock the model to be used in the pipeline
    mock_model = MagicMock()
    mock_model.config.device = config.model.device
    mock_predict_return = torch.rand((1, 6, 6))
    mock_model.predict.return_value = mock_predict_return

    # Patch the _load_model method to return the mocked model
    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):

        pipeline = ObjectDetectionPipeline(mock_model, config)

        mock_image = torch.rand((1, 3, 640, 640))
        preprocessed_image = pipeline.preprocess(mock_image)

        assert preprocessed_image.device.type == model_cfg.device
        assert preprocessed_image.shape == (1, 3, 640, 640)
        assert (preprocessed_image <= 1.0).all()  # Ensure the image is normalized


@pytest.mark.parametrize("use_multithreading", [True, False])
def test_object_detection_pipeline_no_postprocessing(use_multithreading):
    """Test ObjectDetectionPipeline with batchwise NMS enabled."""

    dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
    batch_size = dataset_cfg.batch_size
    mock_model = MagicMock()

    if use_multithreading:
        # Each call returns shape [1, 30240, 6]
        mock_model.side_effect = [torch.rand((1, 30240, 6)) for _ in range(batch_size)]
    else:
        # Single batched output: shape [batch_size, 30240, 6]
        mock_model.return_value = torch.rand((batch_size, 30240, 6))

    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(
            model=model_cfg,
            use_multithreading=use_multithreading,
            apply_nms_batchwise=True,
            apply_length_estimates_batchwise=False,  # Disable length estimation for this test
        )
        mock_model.config.device = config.model.device
        pipeline = ObjectDetectionPipeline(mock_model, config)

        # Mock metadata and dataset required for NMSProcessor
        pipeline.metadata = MagicMock()
        pipeline.metadata.image_meter_width = 1.0
        pipeline.dataset = MagicMock()
        pipeline.dataset.batch_size = batch_size

        # Manually override the dataloader with a mock batch to ensure consistent input shape.
        img_batch = torch.rand(batch_size, 3, 960, 512)
        # shapes format: (image_shape, original_shape) where original_shape = ((h, w), (h, w))
        shapes = [(torch.Size([960, 512]), ((960, 512), (960, 512)))] * batch_size
        pipeline.dataloader = [(img_batch, None, shapes, None)]

        pipeline.model = mock_model

        # Pipeline now returns 4 values: low_preds, high_preds, low_length_estimates, high_length_estimates
        low_preds, high_preds, low_length_estimates, high_length_estimates = pipeline()

        expected_calls = batch_size if use_multithreading else 1
        assert mock_model.call_count == expected_calls

        # Check both outputs are dictionaries
        assert isinstance(low_preds, dict)
        assert isinstance(high_preds, dict)

        # Both should have entries for the batch
        assert len(low_preds) > 0 or len(high_preds) > 0

        # Length estimates should be empty dicts when disabled
        assert isinstance(low_length_estimates, dict)
        assert isinstance(high_length_estimates, dict)


@pytest.mark.skip(
    reason="Postprocessing is deprecated - batchwise NMS is now the standard approach"
)
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
    """Test enabling postprocessing parameters with and without multithreading.

    Note: This test is deprecated. The pipeline now uses batchwise NMS by default,
    which is incompatible with the old postprocessing approach.
    """

    postprocessing_params = {"nms": [NMSConfig(conf=c) for c in confs]}
    dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
    batch_size = dataset_cfg.batch_size

    mock_model = MagicMock()
    if use_multithreading:
        mock_model.side_effect = [torch.rand((1, 30240, 6)) for _ in range(batch_size)]
    else:
        mock_model.return_value = torch.rand((batch_size, 30240, 6))

    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(
            model=model_cfg,
            use_multithreading=use_multithreading,
            apply_nms_batchwise=False,  # Disable batchwise NMS when using postprocessing
        )
        mock_model.config.device = config.model.device
        pipeline = ObjectDetectionPipeline(mock_model, config, postprocessing_params)

        # Manually override the dataloader with a mock batch to ensure consistent input shape.
        img_batch = torch.rand(batch_size, 3, 960, 512)
        shapes = [(torch.Size([960, 512]), ((960, 512), (960, 512)))] * batch_size
        pipeline.dataloader = [(img_batch, None, shapes, None)]

        pipeline.model = mock_model
        pipeline.metadata = MagicMock()
        pipeline.metadata.image_meter_width = 1.0
        pipeline.dataset = MagicMock()
        pipeline.dataset.batch_size = batch_size
        output = pipeline()

        expected_calls = batch_size if use_multithreading else 1
        assert mock_model.call_count == expected_calls

        steps = pipeline.postprocessing_steps
        assert len(output) == len(steps)
        assert len(steps) == len(confs)

        for step, conf_val in zip(steps, confs):
            assert callable(step)
            assert step.keywords["nms_config"].conf == conf_val


@pytest.mark.skip(
    reason="Postprocessing is deprecated - batchwise NMS is now the standard approach"
)
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
    """Test enabling postprocessing parameters.

    Note: This test is deprecated. The pipeline now uses batchwise NMS by default,
    which is incompatible with the old postprocessing approach.
    """

    dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
    batch_size = dataset_cfg.batch_size

    mock_model = MagicMock()
    if use_multithreading:
        mock_model.side_effect = [torch.rand((1, 30240, 6)) for _ in range(batch_size)]
    else:
        mock_model.return_value = torch.rand((batch_size, 30240, 6))

    with patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        dataset_cfg = YOLODatasetConfig(filepath=ARIS_FILE)
        model_cfg = YOLOv5ModelConfig(weights="dummy/path")
        config = ObjectDetectionConfig(
            model=model_cfg,
            use_multithreading=use_multithreading,
            apply_nms_batchwise=False,  # Disable batchwise NMS when using postprocessing
        )
        batch_size = dataset_cfg.batch_size

        mock_model.config.device = config.model.device
        pipeline = ObjectDetectionPipeline(mock_model, config, postprocessing_param)

        # Manually override the dataloader with a mock batch to ensure consistent input shape.
        img_batch = torch.rand(batch_size, 3, 960, 512)
        shapes = [(torch.Size([960, 512]), ((960, 512), (960, 512)))] * batch_size
        pipeline.dataloader = [(img_batch, None, shapes, None)]

        pipeline.model = mock_model
        pipeline.metadata = MagicMock()
        pipeline.metadata.image_meter_width = 1.0
        pipeline.dataset = MagicMock()
        pipeline.dataset.batch_size = batch_size
        output = pipeline()

        expected_calls = batch_size if use_multithreading else 1
        assert mock_model.call_count == expected_calls
        steps = pipeline.postprocessing_steps

        if steps:
            assert len(output) == len(steps)


@pytest.mark.skip(
    reason="Postprocessing is deprecated - batchwise NMS is now the standard approach"
)
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
            ObjectDetectionPipeline(mock_model, config, processing_params)


def test_safe_execution_eventually_succeeds():
    call_tracker = {"count": 0}

    @safe_execution(default_return="FAILED", max_retries=3, delay=0.01)
    def flaky_function():
        call_tracker["count"] += 1
        if call_tracker["count"] < 3:
            raise ValueError("Temporary failure")
        return "SUCCESS"

    result = flaky_function()

    assert result == "SUCCESS"
    assert call_tracker["count"] == 3


def test_safe_execution_fails_all_retries():
    call_tracker = {"count": 0}

    @safe_execution(default_return=[], max_retries=2, delay=0.01)
    def always_fails():
        call_tracker["count"] += 1
        raise RuntimeError("Still broken")

    result = always_fails()

    assert result == []
    assert call_tracker["count"] == 2


@patch("time.sleep", return_value=None)
def test_exponential_backoff_sleep_called(mock_sleep):
    call_tracker = {"count": 0}

    @safe_execution(default_return=[], max_retries=3, delay=0.1)
    def fails_then_succeeds():
        call_tracker["count"] += 1
        if call_tracker["count"] < 3:
            raise Exception("try again")
        return "OK"

    result = fails_then_succeeds()
    assert result == "OK"
    assert mock_sleep.call_count == 2
    assert mock_sleep.call_args_list[0][0][0] == 0.1  # 1st backoff
    assert mock_sleep.call_args_list[1][0][0] == 0.2  # 2nd backoff (0.1 * 2^1)


def test_ignore_hidden_files(tmp_path):
    # Create visible and hidden files
    (tmp_path / "visible_file.aris").touch()
    (tmp_path / ".hidden_file.aris").touch()
    hidden_subdir = tmp_path / ".hidden_dir"
    hidden_subdir.mkdir()
    (hidden_subdir / "another_hidden_file.aris").touch()

    results = get_all_valid_files_in_dir(tmp_path)
    result_names = [f.name for f in results]

    assert "visible_file.aris" in result_names
    assert ".hidden_file.aris" not in result_names
    assert "another_hidden_file.aris" not in result_names


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.config.device = "cpu"
    # Mock return value for inference: [batch_size, num_preds, 6]
    model.return_value = torch.rand((1, 10, 6))
    return model


@pytest.fixture
def mock_length_estimator():
    estimator = MagicMock()
    # Mock run return value
    estimator.run.return_value = {(0, 0): {"length": 10.0}}
    return estimator


def test_object_detection_pipeline_with_length(mock_model, mock_length_estimator):
    """Test ObjectDetectionPipeline with length estimation enabled."""

    config = ObjectDetectionConfig(
        model=YOLOv5ModelConfig(weights="dummy"),
        apply_length_estimates_batchwise=True,
        apply_nms_batchwise=False,
        use_multithreading=False,
    )

    with patch(
        "fisheye.pipelines.detection.create_length_estimator",
        return_value=mock_length_estimator,
    ) as mock_create_le, patch(
        "fisheye.pipelines.detection.get_length_model_config"
    ) as mock_get_config, patch.object(
        YOLOv5ObjectDetectionModel, "_load_model", return_value=mock_model
    ):
        pipeline = ObjectDetectionPipeline(mock_model, config)

        # Mock dataset/dataloader
        pipeline.dataset = MagicMock()
        pipeline.dataset.batch_size = 1
        pipeline.metadata = MagicMock()

        # Mock dataloader: yield (img, None, shapes, original_img)
        img = torch.rand(1, 3, 640, 640)
        original_img = torch.rand(1, 3, 1080, 1920)
        shapes = [(torch.Size([640, 640]), ((1080, 1920), (1080, 1920)))]
        pipeline.dataloader = [(img, None, shapes, original_img)]

        # Run pipeline
        _, _, low_lengths, high_lengths = pipeline()

        # Verify length estimator was created and called
        mock_create_le.assert_called_once()
        assert (
            mock_length_estimator.run.call_count >= 1
        )  # Called for low and high preds

        # Verify output contains length estimates
        assert len(low_lengths) > 0 or len(high_lengths) > 0


def test_detect_track_count_pipeline_integration():
    """Test DetectTrackCountPipeline integration with length estimates."""

    # Mock detection pipeline
    detect_pipe = MagicMock()
    detect_pipe.metadata.image_meter_width = 1.0
    detect_pipe.metadata.image_meter_height = 1.0
    detect_pipe.metadata.xdim = 1920
    detect_pipe.metadata.ydim = 1080
    detect_pipe.apply_length_estimates_batchwise = True

    # Mock detection output
    # low_preds, high_preds, low_lengths, high_lengths
    detect_pipe.return_value = (
        {(0, 0): torch.tensor([[100, 100, 200, 200, 0.9, 0]])},  # low_preds
        {},  # high_preds
        {
            (0, 0): {0: {"pred_kpts_global_px": [[100, 100], [200, 200]]}}
        },  # low_lengths (mock format)
        {},  # high_lengths
    )

    pipeline = DetectTrackCountPipeline(detect_pipe=detect_pipe)

    with patch("fisheye.pipelines.pipeline.run_tracker") as mock_run_tracker, patch(
        "fisheye.pipelines.pipeline.LengthProcessor"
    ) as MockLengthProcessor, patch(
        "fisheye.pipelines.pipeline.Count"
    ) as MockCount, patch(
        "fisheye.pipelines.pipeline.save_to_disk"
    ), patch(
        "fisheye.pipelines.pipeline.asdict"
    ) as mock_asdict:

        mock_tracker_output = TrackerOutput(
            start_frame=0,
            end_frame=1,
            image_meter_width=1.0,
            image_meter_height=1.0,
            frames=[],
            metadata=[],
        )
        mock_run_tracker.return_value = mock_tracker_output

        # Return a dict representation of the tracker output
        mock_asdict.return_value = {
            "frames": [],
            "start_frame": 0,
            "end_frame": 1,
            "image_meter_width": 1.0,
            "image_meter_height": 1.0,
            "metadata": [],
        }

        # Set up mock length processor
        mock_processor = MockLengthProcessor.return_value
        mock_processor.process_from_tracks.return_value = {
            1: {"filtered_lengths_cm": 15.0}  # fish_id 1
        }

        MockCount.return_value.count.return_value = (
            (1, 0),
            {"left": [(1, 0, [100, 100, 100, 100])], "right": []},
        )

        pipeline._run(file="dummy.aris", output_dir=".")

        # Verify length processor was called
        mock_processor.process_from_tracks.assert_called_once()


def test_formatted_crossings_frame_fallback():
    """Test that Frame# falls back to crossing frame when frame_id_closest_to_mean is missing.

    Specific testing this logic in pipelines/pipeline.py:
        "Frame#": len_outputs.get(track_id, {}).get("frame_id_closest_to_mean", frame)

    Three scenarios:
    1. len_outputs has frame_id_closest_to_mean -> use it
    2. len_outputs missing track_id entirely -> fall back to crossing frame
    3. len_outputs has track_id but missing frame_id_closest_to_mean -> fall back to crossing frame
    """

    # Mock detection pipeline
    detect_pipe = MagicMock()
    detect_pipe.metadata.image_meter_width = 1.0
    detect_pipe.metadata.image_meter_height = 1.0
    detect_pipe.metadata.xdim = 1920
    detect_pipe.metadata.ydim = 1080
    detect_pipe.apply_length_estimates_batchwise = True

    # Mock detection output
    detect_pipe.return_value = (
        {(0, 0): torch.tensor([[100, 100, 200, 200, 0.9, 0]])},  # low_preds
        {},  # high_preds
        {(0, 0): {0: {"pred_kpts_global_px": [[100, 100], [200, 200]]}}},  # low_lengths
        {},  # high_lengths
    )

    pipeline = DetectTrackCountPipeline(detect_pipe=detect_pipe)

    with patch.object(pipeline, "_estimate_lengths") as mock_estimate_lengths, patch(
        "fisheye.pipelines.pipeline.run_tracker"
    ) as mock_run_tracker, patch(
        "fisheye.pipelines.pipeline.Count"
    ) as MockCount, patch(
        "fisheye.pipelines.pipeline.save_to_disk"
    ), patch(
        "fisheye.pipelines.pipeline.asdict"
    ) as mock_asdict, patch(
        "fisheye.pipelines.pipeline.tracker_output_to_dict_rows"
    ) as mock_tracker_to_dict:

        # Set up the len_outputs with our three test scenarios
        mock_estimate_lengths.return_value = {
            # Scenario 1: Has frame_id_closest_to_mean (should use 42)
            1: {
                "filtered_lengths_cm": 15.0,
                "frame_id_closest_to_mean": 42,
                "global_coords_px": [[100, 100], [200, 200]],
            },
            # Scenario 2: Missing frame_id_closest_to_mean (should fall back to crossing frame)
            2: {
                "filtered_lengths_cm": 20.0,
                "global_coords_px": [[150, 150], [250, 250]],
            },
            # Scenario 3: track_id 3 is completely missing from len_outputs
        }

        mock_tracker_output = TrackerOutput(
            start_frame=0,
            end_frame=1,
            image_meter_width=1.0,
            image_meter_height=1.0,
            frames=[],
            metadata=[],
        )
        mock_run_tracker.return_value = mock_tracker_output

        # Return a dict representation of the tracker output
        mock_asdict.return_value = {
            "frames": [],
            "start_frame": 0,
            "end_frame": 1,
            "image_meter_width": 1.0,
            "image_meter_height": 1.0,
            "metadata": [],
        }

        # Mock tracker output to dict rows
        mock_tracker_to_dict.return_value = []

        # Mock crossing frames with different track IDs and frame numbers
        MockCount.return_value.count.return_value = (
            (3, 0),  # 3 left crossings, 0 right
            {
                "left": [
                    (1, 10, [100, 100, 100, 100]),  # track_id=1, frame=10
                    (2, 20, [150, 150, 100, 100]),  # track_id=2, frame=20
                    (
                        3,
                        30,
                        [200, 200, 100, 100],
                    ),  # track_id=3, frame=30 (not in len_outputs)
                ],
                "right": [],
            },
        )

        result = pipeline._run(file="dummy.aris", output_dir=".")

        # Verify we got 3 crossings
        assert len(result) == 3

        # Scenario 1: track_id=1 should use frame_id_closest_to_mean=42
        crossing_1 = next(c for c in result if c["ID"] == 1)
        assert crossing_1["Frame#"] == 42, (
            f"Expected Frame# to be 42 (from frame_id_closest_to_mean), "
            f"but got {crossing_1['Frame#']}"
        )

        # Scenario 2: track_id=2 should fall back to crossing frame=20
        crossing_2 = next(c for c in result if c["ID"] == 2)
        assert crossing_2["Frame#"] == 20, (
            f"Expected Frame# to fall back to crossing frame 20, "
            f"but got {crossing_2['Frame#']}"
        )

        # Scenario 3: track_id=3 should fall back to crossing frame=30
        crossing_3 = next(c for c in result if c["ID"] == 3)
        assert crossing_3["Frame#"] == 30, (
            f"Expected Frame# to fall back to crossing frame 30, "
            f"but got {crossing_3['Frame#']}"
        )

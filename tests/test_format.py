import pytest
from unittest.mock import MagicMock

from fisheye.format import format_single_crossing


@pytest.fixture
def mock_metadata():
    """Create mock metadata object."""
    metadata = MagicMock()
    metadata.image_meter_width = 10.0
    metadata.image_meter_height = 20.0
    return metadata


@pytest.fixture
def sample_len_outputs():
    """Sample length estimation outputs."""
    return {
        1: {
            "frame_id_closest_to_mean": 105,
            "filtered_lengths_cm": 30.5,
            "global_coords_px": [[100, 200], [150, 250]],
        },
        2: {
            "frame_id_closest_to_mean": 200,
            "filtered_lengths_cm": 25.0,
            "global_coords_px": [[300, 400], [350, 450]],
        },
        3: {
            "frame_id_closest_to_mean": None,  # No valid estimate
            "filtered_lengths_cm": None,
            "global_coords_px": None,
        },
    }


class TestFormatCrossingEvent:
    """Test format_crossing_event function."""

    def test_empty_crossing_minimal_args(self, mock_metadata):
        """Test formatting empty crossing with only required args."""
        result = format_single_crossing("test_file.aris", mock_metadata)

        assert result["Source.Name"] == "test_file.aris"
        assert result["Frame#"] is None
        assert result["Dir"] is None
        assert result["ID"] is None
        assert result["bbox"] is None
        assert result["metadata"] == mock_metadata
        assert result["global_coords_px"] is None
        assert result["L(cm)"] is None

    def test_valid_crossing_left_upstream_left(self, mock_metadata, sample_len_outputs):
        """Test valid crossing: fish crosses left, upstream is left -> Up."""
        result = format_single_crossing(
            filename="test_file.aris",
            metadata=mock_metadata,
            track_id=1,
            frame=100,
            bbox=[0.5, 0.5, 0.1, 0.1],
            upstream_direction="left",
            direction="left",
            len_outputs=sample_len_outputs,
        )

        assert result["Source.Name"] == "test_file.aris"
        assert result["Frame#"] == 105  # Uses closest frame from len_outputs
        assert result["Dir"] == "Up"  # upstream == crossing_side
        assert result["ID"] == 1
        assert result["bbox"] == [0.5, 0.5, 0.1, 0.1]
        assert result["metadata"] == mock_metadata
        assert result["global_coords_px"] == [[100, 200], [150, 250]]
        assert result["L(cm)"] == 30.5

    def test_valid_crossing_left_upstream_right(
        self, mock_metadata, sample_len_outputs
    ):
        """Test valid crossing: fish crosses left, upstream is right -> Down."""
        result = format_single_crossing(
            filename="test_file.aris",
            metadata=mock_metadata,
            track_id=2,
            frame=195,
            bbox=[0.6, 0.6, 0.15, 0.15],
            upstream_direction="right",
            direction="left",
            len_outputs=sample_len_outputs,
        )

        assert result["Frame#"] == 200
        assert result["Dir"] == "Down"
        assert result["ID"] == 2
        assert result["L(cm)"] == 25.0

    def test_crossing_without_length_estimate(self, mock_metadata):
        """Test crossing when fish has no length estimate."""
        len_outputs = {}  # Empty - no estimate for this fish

        result = format_single_crossing(
            filename="test_file.aris",
            metadata=mock_metadata,
            track_id=99,
            frame=500,
            bbox=[0.7, 0.7, 0.2, 0.2],
            upstream_direction="left",
            direction="right",
            len_outputs=len_outputs,
        )

        assert result["Frame#"] == 500  # Falls back to provided frame (from Counter)
        assert result["Dir"] == "Down"
        assert result["ID"] == 99
        assert result["global_coords_px"] is None
        assert result["L(cm)"] == 0.0  # Uses FC_SCHEMA default

    def test_crossing_with_none_length_estimate(
        self, mock_metadata, sample_len_outputs
    ):
        """Test crossing when length estimate exists but is None."""
        result = format_single_crossing(
            filename="test_file.aris",
            metadata=mock_metadata,
            track_id=3,  # Has None values in sample_len_outputs
            frame=300,
            bbox=[0.8, 0.8, 0.1, 0.1],
            upstream_direction="left",
            direction="left",
            len_outputs=sample_len_outputs,
        )

        assert result["Frame#"] == 300
        assert result["global_coords_px"] is None
        assert result["L(cm)"] == 0.0

    def test_crossing_with_zero_length(self, mock_metadata):
        """Test crossing when filtered_lengths_cm is 0 (edge case)."""
        len_outputs = {
            1: {
                "frame_id_closest_to_mean": 100,
                "filtered_lengths_cm": 0,  # Zero length
                "global_coords_px": [[10, 20], [30, 40]],
            }
        }

        result = format_single_crossing(
            filename="test_file.aris",
            metadata=mock_metadata,
            track_id=1,
            frame=100,
            bbox=[0.5, 0.5, 0.1, 0.1],
            upstream_direction="left",
            direction="left",
            len_outputs=len_outputs,
        )

        assert result["L(cm)"] == 0.0

    def test_crossing_none_len_outputs(self, mock_metadata):
        """Test crossing when len_outputs is None."""
        result = format_single_crossing(
            filename="test_file.aris",
            metadata=mock_metadata,
            track_id=1,
            frame=100,
            bbox=[0.5, 0.5, 0.1, 0.1],
            upstream_direction="left",
            direction="left",
            len_outputs=None,
        )

        assert result["Frame#"] == 100
        assert result["global_coords_px"] is None
        assert result["L(cm)"] == 0.0

    def test_all_direction_combinations(self, mock_metadata, sample_len_outputs):
        """Test all combinations of upstream_direction and crossing_side."""
        test_cases = [
            ("left", "left", "Up"),
            ("left", "right", "Down"),
            ("right", "left", "Down"),
            ("right", "right", "Up"),
        ]

        for upstream, crossing, expected_dir in test_cases:
            result = format_single_crossing(
                filename="test.aris",
                metadata=mock_metadata,
                track_id=1,
                frame=100,
                bbox=[0.5, 0.5, 0.1, 0.1],
                upstream_direction=upstream,
                direction=crossing,
                len_outputs=sample_len_outputs,
            )

            assert result["Dir"] == expected_dir, (
                f"Failed for upstream={upstream}, crossing={crossing}: "
                f"expected {expected_dir}, got {result['Dir']}"
            )

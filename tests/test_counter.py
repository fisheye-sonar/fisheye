import pytest
from fisheye.count.counter import Count


@pytest.fixture
def loi_counter():
    """Fixture for creating a Count instance with 'loi' protocol."""
    return Count(protocol="loi")


def test_invalid_counter_type(tracks_crossing_left_to_right):
    """Test code will not run with an invalid counter."""
    with pytest.raises(ValueError, match="Protocol '.*' is not supported."):
        Count(protocol="invalid_protocol").count(tracks_crossing_left_to_right)


@pytest.mark.parametrize(
    "tracks, expected_left, expected_right",
    [
        ("tracks_crossing_left_to_right", 0, 1),  # Left-to-right crossing
        ("no_cross_tracks", 0, 0),  # No crossings
        ("no_tracks_null_dict", 0, 0),  # No tracks
        ("no_tracks_empty_dict", 0, 0),  # No tracks different structure
        ("milling", 0, 0),  # Milling behavior
        ("tracks_crossing_right_to_left", 1, 0),  # Right-to-left crossing
    ],
)
def test_loi_counter_counts(
    loi_counter, request, tracks, expected_left, expected_right
):
    """Test counting code with different track scenarios."""
    track_data = request.getfixturevalue(tracks)
    (left_count, right_count), crossings = loi_counter.count(track_data)

    assert left_count == expected_left
    assert right_count == expected_right

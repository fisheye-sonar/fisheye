from fisheye.count.counter import Count


def test_loi_counter_count(sample_tracks):
    """Test counting fish that cross the line."""
    counter = Count(protocol="LOI")
    left_count, right_count = counter.count(sample_tracks)

    assert left_count == 0
    assert right_count == 1


def test_loi_counter_no_crossings(no_cross_tracks):
    """Test counting fish that cross the line."""
    counter = Count(protocol="LOI")
    left_count, right_count = counter.count(no_cross_tracks)

    assert left_count == 0
    assert right_count == 0

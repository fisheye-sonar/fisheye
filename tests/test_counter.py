from fisheye.count.counter import Count


def test_loi_counter_count(sample_tracks):
    """Test counting fish that cross the line."""
    counter = Count(protocol="loi")
    (left_count, right_count), crossings = counter.count(sample_tracks)

    assert left_count == 0
    assert right_count == 1


def test_loi_counter_no_crossings(no_cross_tracks):
    """Test counting fish that cross the line."""
    counter = Count(protocol="loi")
    (left_count, right_count), crossings = counter.count(no_cross_tracks)

    assert left_count == 0
    assert right_count == 0


def test_loi_counter_no_tracks(no_tracks):
    """Test counting when there are no tracks."""
    counter = Count(protocol="loi")
    (left_count, right_count), crossings = counter.count(no_tracks)

    assert left_count == 0
    assert right_count == 0

from enum import Enum


class TrackingMethod(Enum):
    """Types of tracking algorithms currently supported."""

    NONE = 0
    CONF_BOOST = 1
    BYTETRACK = 2
    SORT = 3

    def toString(val):
        if val == TrackingMethod.NONE:
            return "None"
        if val == TrackingMethod.CONF_BOOST:
            return "Confidence Boost"
        if val == TrackingMethod.BYTETRACK:
            return "ByteTrack"
        if val == TrackingMethod.SORT:
            return "Sort"

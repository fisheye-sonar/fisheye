from enum import Enum


class TrackingMethod(str, Enum):
    """Types of tracking algorithms currently supported."""

    NONE = "none"
    CONF_BOOST = "conf_boost"
    BYTETRACK = "bytetrack"
    SORT = "sort"


class CountingMethod(str, Enum):
    NONE = "none"
    LOI = "loi"

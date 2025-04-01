from enum import Enum


class TrackingMethod(str, Enum):
    """Types of tracking algorithms currently supported."""

    NONE = "none"
    CONF_BOOST = "conf_boost"
    BYTETRACK = "bytetrack"
    SORT = "sort"


class CountingMethod(str, Enum):
    """Types of counting algorithms currently supported."""

    NONE = "none"
    LOI = "loi"


class ExportType(str, Enum):
    """Types of exports currently supported."""

    NONE = "none"
    CSV = "csv"

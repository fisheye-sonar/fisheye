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
    """Types of exports/formats currently supported."""

    NONE = "none"
    CSV = "csv"
    TXT = "txt"
    MOT = "mot"


class DeviceType(str, Enum):
    """Types of devices currently supported."""

    GPU = "gpu"
    CPU = "cpu"
    MPS = "mps"  # enable GPU on Macs with Apple Silicon

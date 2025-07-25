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
    SUMMARY_CSV = "summary_csv"
    DETAILED_CSV = "detailed_csv"
    TXT = "txt"
    MOT = "mot"


class DeviceType(str, Enum):
    """Types of devices currently supported."""

    CUDA = "cuda:0"  # single NVIDIA GPU
    CPU = "cpu"
    MPS = "mps"  # enable GPU on Macs with Apple Silicon


class ValidExtensions(Enum):
    ARIS = ".aris"
    DDF = ".ddf"


class UpstreamDirectionTypes(str, Enum):
    """Types of directions currently supported."""

    LEFT = "left"
    RIGHT = "right"


class IgnoredSystemDirs(str, Enum):
    RECYCLE_BIN = "$RECYCLE.BIN"
    SYSTEM_VOLUME = "System Volume Information"
    SPOTLIGHT = ".Spotlight-V100"
    FSEVENTSD = ".fseventsd"
    TRASH = ".Trash-1000"
    TEMP_ITEMS = ".TemporaryItems"


class IgnoredFilePrefixes(str, Enum):
    DS_STORE = ".DS_Store"
    APPLE_RESOURCE = "._"
    THUMBS_DB = "Thumbs.db"
    VOLUME_ICON = ".VolumeIcon.icns"


IGNORED_DIR_NAMES = {e.value for e in IgnoredSystemDirs}
IGNORED_FILE_PREFIXES = {e.value for e in IgnoredFilePrefixes}

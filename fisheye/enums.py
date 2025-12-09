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
    FC = "fc"
    MOT = "mot"


class DeviceType(str, Enum):
    """Types of devices currently supported."""

    CUDA = "cuda:0"  # single NVIDIA GPU
    CPU = "cpu"
    MPS = "mps"  # enable GPU on Macs with Apple Silicon


class ValidExtensions(Enum):
    """Types of valid file extensions currently supported."""

    ARIS = ".aris"
    DDF = ".ddf"


class UpstreamDirectionTypes(str, Enum):
    """Types of directions currently supported."""

    LEFT = "left"
    RIGHT = "right"


class IgnoredSystemDirs(str, Enum):
    """Types of system-level directories we do not support.

    Already check for dot-prefixed values in get_all_valid_files_in_dir().
    """

    RECYCLE_BIN = "$RECYCLE.BIN"
    SYSTEM_VOLUME = "System Volume Information"
    VMSNAPSHOTS = "vm_snapshots"
    CONFIG_MSI = "Config.Msi"
    RECOVERY = "Recovery"
    LOST_FOUND = "lost+found"
    TRASH_1000 = "Trash-1000"


class IgnoredFilePrefixes(str, Enum):
    """Types of files we do not support.

    Already check for dot-prefixed values in get_all_valid_files_in_dir().
    """

    THUMBS_DB = "Thumbs.db"
    EHTHUMBS = "ehthumbs.db"
    DESKTOP_INI = "desktop.ini"


class DetectorType(Enum):
    """Types of detectors currently supported."""

    YOLOv5 = "yolov5"
    YOLOv11 = "yolov11"


class LengthEstimatorType(Enum):
    """Types of length estimators currently supported."""

    UNET = "unet"
    HEATMAP_CNN = "heatmap_cnn"


IGNORED_DIR_NAMES = {e.value for e in IgnoredSystemDirs}
IGNORED_FILE_PREFIXES = {e.value for e in IgnoredFilePrefixes}

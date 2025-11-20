from enum import Enum


class DatasetFormat(Enum):
    """Types of dataset currently supported."""

    YOLO = "yolo"
    COCO = "coco"

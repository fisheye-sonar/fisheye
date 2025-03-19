from abc import ABC, abstractmethod

from fisheye.enums import TrackingMethod


class BaseTracker(ABC):
    """
    Base class for Multi-Object Trackers (MOT) such as ByteTrack and SORT.

    This class initializes common parameters and handles basic tracking attributes.

    Args:
        max_age (int): Maximum age for a track to be kept without updates.
        min_hits (int): Minimum number of hits required to confirm a track.
        iou_threshold (float): IOU threshold for association.
    """

    type = TrackingMethod.NONE

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initializes common tracker parameters.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    @abstractmethod
    def update(self):
        """Abstract method to update the tracker with new detections."""
        pass

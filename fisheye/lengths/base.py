from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseLengthEstimator(ABC):
    """Abstract base class for length estimators."""

    def __init__(self, metadata: Any):
        self.metadata = metadata

    @abstractmethod
    def run(self, frames_batch, pred_bboxes) -> Dict[int, Any]:
        """
        Run length estimation on a batch of frames.

        Args:
            frames_batch: Batch of frames.
            pred_bboxes: Predicted bounding boxes.

        Returns:
            Dict mapping frame number to length estimation data.
        """
        pass

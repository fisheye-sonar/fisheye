from abc import abstractmethod, ABC

import pandas as pd

from fisheye.enums import CountingMethod


class BaseCounter(ABC):
    type = CountingMethod.NONE
    """Abstract base class for counting methods."""

    @abstractmethod
    def count(self, mot_df: pd.DataFrame, line: float = 0.5):
        """Count fish based on the specified method."""
        pass

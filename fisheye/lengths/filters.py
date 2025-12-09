"""Length estimation filtering and validation.

This module provides filtering logic for length estimates based on various
quality criteria such as edge distance, length stability, and velocity consistency.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from fisheye.configs.inference import LengthEstimationConfig
from fisheye.lengths.measure_utils import (
    get_min_edge_distances_pxl,
    get_change_in_length,
    get_velocity_dev,
    calc_len,
)


@dataclass
class FilterResult:
    """Results from applying filters to length estimates."""

    masks: Dict[str, List[bool]]  # Individual filter masks
    combined_mask: np.ndarray  # Combined boolean mask
    num_passed: int  # Number of estimates that passed all filters
    mean_length_cm: float  # Mean of all lengths
    filtered_mean_length_cm: float  # Mean of filtered lengths (or NaN if none passed)
    closest_to_mean_idx: Optional[int]  # Index of estimate closest to filtered mean


class LengthFilter:
    """Filters length estimates based on quality criteria.

    Applies configurable filters for:
    - Edge distance (proximity to cone boundaries)
    - Length stability (deviation from moving average)
    - Velocity consistency (deviation from expected motion)
    """

    def __init__(self, config: LengthEstimationConfig, metadata):
        """Initialize the filter with configuration.

        Args:
            config: Length estimation configuration with filter thresholds
            metadata: Video metadata containing pixel-to-meter conversion
        """
        self.config = config
        self.metadata = metadata

    def apply_filters(
        self,
        pred_kpts_global_px: List[np.ndarray],
        frame_nums: List[int],
        cone_params: Tuple[float, float, float, float],
    ) -> FilterResult:
        """Apply all configured filters to length estimates for a single fish.

        Args:
            pred_kpts_global_px: List of keypoint pairs [(x0,y0), (x1,y1)] in global pixel coords
            frame_nums: Corresponding frame numbers
            cone_params: (ml, bl, mr, br) - left and right cone edge parameters

        Returns:
            FilterResult containing masks and statistics
        """
        masks = {}
        masks["all"] = [True] * len(pred_kpts_global_px)

        # Convert to centimeters for filtering
        pred_kpts_global_0_cm = [
            kpts[0] * self.metadata.pixel_meter_size * 100
            for kpts in pred_kpts_global_px
        ]
        pred_kpts_global_1_cm = [
            kpts[1] * self.metadata.pixel_meter_size * 100
            for kpts in pred_kpts_global_px
        ]
        pred_lens_cm = [
            calc_len(kpts) * self.metadata.pixel_meter_size * 100
            for kpts in pred_kpts_global_px
        ]

        # Filter 1: Edge distance
        if self.config.min_edge_dist_tolerance_px is not None:
            masks["edge_dist"] = self._filter_edge_distance(
                pred_kpts_global_px, cone_params
            )

        # Filter 2: Length stability
        if self.config.length_delta_tolerance_cm is not None:
            masks["length"] = self._filter_length_stability(
                pred_kpts_global_0_cm, pred_kpts_global_1_cm, frame_nums
            )

        # Filter 3: Velocity consistency
        if self.config.vel_delta_tolerance is not None:
            masks["velocity"] = self._filter_velocity_consistency(
                pred_kpts_global_0_cm, pred_kpts_global_1_cm, frame_nums
            )

        # Combine all masks
        combined_mask = np.all(np.array([v for v in masks.values()]) == 1, axis=0)
        num_passed = int(np.sum(combined_mask))

        # Calculate statistics
        mean_length_cm = float(np.mean(pred_lens_cm))

        if num_passed == 0:
            filtered_mean_length_cm = np.nan
            closest_to_mean_idx = None
        else:
            filtered_lengths = [
                pred_lens_cm[i] for i in range(len(pred_lens_cm)) if combined_mask[i]
            ]
            filtered_mean_length_cm = float(np.mean(filtered_lengths))

            # Find index closest to filtered mean
            pred_lens_filtered = pred_lens_cm * combined_mask
            pred_lens_filtered_mean = np.mean(
                pred_lens_filtered[pred_lens_filtered != 0]
            )
            closest_idx = int(
                np.argmin(abs(pred_lens_filtered - pred_lens_filtered_mean))
            )
            closest_to_mean_idx = closest_idx if combined_mask[closest_idx] else None

        return FilterResult(
            masks=masks,
            combined_mask=combined_mask,
            num_passed=num_passed,
            mean_length_cm=mean_length_cm,
            filtered_mean_length_cm=filtered_mean_length_cm,
            closest_to_mean_idx=closest_to_mean_idx,
        )

    def _filter_edge_distance(
        self,
        pred_kpts_global_px: List[np.ndarray],
        cone_params: Tuple[float, float, float, float],
    ) -> List[bool]:
        """Filter based on minimum distance from cone edges.

        Args:
            pred_kpts_global_px: Keypoint pairs in pixel coordinates
            cone_params: (ml, bl, mr, br) cone edge line parameters

        Returns:
            Boolean mask indicating which estimates pass the filter
        """
        ml, bl, mr, br = cone_params
        min_edge_distances_px = []

        for kpts in pred_kpts_global_px:
            edge_dist = get_min_edge_distances_pxl(kpts, ml, bl, mr, br)[0]
            min_edge_distances_px.append(edge_dist)

        return [
            dist >= self.config.min_edge_dist_tolerance_px
            for dist in min_edge_distances_px
        ]

    def _filter_length_stability(
        self,
        pred_kpts_global_0_cm: List[float],
        pred_kpts_global_1_cm: List[float],
        frame_nums: List[int],
    ) -> List[bool]:
        """Filter based on deviation from moving average length.

        Args:
            pred_kpts_global_0_cm: First keypoint coordinates in cm
            pred_kpts_global_1_cm: Second keypoint coordinates in cm
            frame_nums: Frame numbers

        Returns:
            Boolean mask indicating which estimates pass the filter
        """
        change_in_length, _moving_avg = get_change_in_length(
            pred_kpts_global_0_cm,
            pred_kpts_global_1_cm,
            frame_nums,
            window_size=self.config.length_window_size,
            robust=True,
        )

        return [
            abs(change) < self.config.length_delta_tolerance_cm
            for change in change_in_length
        ]

    def _filter_velocity_consistency(
        self,
        pred_kpts_global_0_cm: List[float],
        pred_kpts_global_1_cm: List[float],
        frame_nums: List[int],
    ) -> List[bool]:
        """Filter based on velocity deviation.

        Args:
            pred_kpts_global_0_cm: First keypoint coordinates in cm
            pred_kpts_global_1_cm: Second keypoint coordinates in cm
            frame_nums: Frame numbers

        Returns:
            Boolean mask indicating which estimates pass the filter
        """
        velocity_deviations = get_velocity_dev(
            pred_kpts_global_0_cm,
            pred_kpts_global_1_cm,
            frame_nums,
            window_size=self.config.vel_window_size,
        )

        return [vel < self.config.vel_delta_tolerance for vel in velocity_deviations]

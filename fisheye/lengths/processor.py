"""High-level length estimation processor.

This module coordinates length estimation, filtering, and result aggregation
for tracked fish across video frames.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import structlog

from fisheye.configs.inference import LengthEstimationConfig
from fisheye.lengths.filters import LengthFilter
from fisheye.lengths.measure_utils import get_cone_edges

logger = structlog.get_logger(__name__)


@dataclass
class LengthEstimate:
    """Length estimate result for a single fish."""

    fish_id: int
    all_lengths_cm: float  # Mean of all raw length estimates
    filtered_lengths_cm: float  # Mean of filtered estimates (NaN if none passed)
    num_filtered: int  # Number of estimates that passed filters
    frame_id_closest_to_mean: Optional[int]  # Frame with estimate closest to mean
    coords_px: Optional[List]  # Keypoint coordinates for best estimate


class LengthProcessor:
    """Processes length estimates from tracking data with quality filtering.

    This class coordinates:
    1. Aggregating length estimates across frames for each tracked fish
    2. Applying quality filters (edge distance, stability, velocity)
    3. Computing final length estimates
    """

    def __init__(self, config: LengthEstimationConfig, metadata):
        """Initialize the processor.

        Args:
            config: Length estimation configuration
            metadata: Video metadata (pixel size, dimensions, etc.)
        """
        self.config = config
        self.metadata = metadata
        self.filter = LengthFilter(config, metadata)

        # Compute cone parameters once
        (
            _cone_points_left,
            _cone_points_right,
            cone_eq_params_left,
            cone_eq_params_right,
        ) = get_cone_edges(metadata)
        self.cone_params = (
            cone_eq_params_left[0],  # ml
            cone_eq_params_left[1],  # bl
            cone_eq_params_right[0],  # mr
            cone_eq_params_right[1],  # br
        )

    def process_from_tracks(
        self,
        frames_preds: List[Dict],
        low_length_estimates: Optional[Dict] = None,
        high_length_estimates: Optional[Dict] = None,
        batch_size: int = 1,
    ) -> Dict[int, LengthEstimate]:
        """Process length estimates from tracking predictions.

        Args:
            frames_preds: List of frame predictions from tracker
            low_length_estimates: Optional dict of low-confidence length estimates
            high_length_estimates: Optional dict of high-confidence length estimates
            batch_size: Batch size used during detection

        Returns:
            Dictionary mapping fish_id to LengthEstimate
        """
        # First, associate length estimates with tracked fish
        if low_length_estimates is not None and high_length_estimates is not None:
            self._associate_length_estimates(
                frames_preds,
                low_length_estimates,
                high_length_estimates,
                batch_size,
            )

        # Aggregate all length estimates per fish
        all_length_estimates = self._aggregate_estimates_by_fish(frames_preds)

        # Process each fish
        len_outputs = {}
        for fish_id, data in all_length_estimates.items():
            len_outputs[fish_id] = self._process_single_fish(fish_id, data)

        return len_outputs

    def _associate_length_estimates(
        self,
        frames_preds: List[Dict],
        low_length_estimates: Dict,
        high_length_estimates: Dict,
        batch_size: int,
    ):
        """Associate length estimates with tracked fish in frames_preds.

        Modifies frames_preds in-place to add 'length_estimate' field to each fish.
        """
        for frame_pred in frames_preds:
            frame_num = frame_pred["frame_num"]
            frame_bs_num = (
                int(frame_num / batch_size),
                frame_num % batch_size,
            )

            for fish_pred in frame_pred["fish"]:
                bbox_index = fish_pred["bbox_index"]
                det_index = fish_pred["det_index"]

                if det_index == 0:
                    matched_estimate = low_length_estimates[frame_bs_num][bbox_index]
                else:
                    # High confidence detection
                    matched_estimate = high_length_estimates[frame_bs_num][bbox_index]

                fish_pred["length_estimate"] = matched_estimate

    def _aggregate_estimates_by_fish(self, frames_preds: List[Dict]) -> Dict[int, Dict]:
        """Aggregate all length estimates for each tracked fish.

        Args:
            frames_preds: Frame predictions with length estimates

        Returns:
            Dict mapping fish_id to {frame_num: [...], pred_kpts_global_px: [...]}
        """
        all_estimates = {}

        for frame_pred in frames_preds:
            frame_num = frame_pred["frame_num"]

            for fish_pred in frame_pred["fish"]:
                fish_id = fish_pred["id"]

                # Initialize if first time seeing this fish
                if fish_id not in all_estimates:
                    all_estimates[fish_id] = {
                        "frame_num": [],
                        "pred_kpts_global_px": [],
                    }

                # Add this frame's estimate
                length_estimate = fish_pred.get("length_estimate", {})
                if "pred_kpts_global_px" in length_estimate:
                    all_estimates[fish_id]["frame_num"].append(frame_num)
                    all_estimates[fish_id]["pred_kpts_global_px"].append(
                        length_estimate["pred_kpts_global_px"]
                    )

        return all_estimates

    def _process_single_fish(
        self, fish_id: int, data: Dict
    ) -> Optional[LengthEstimate]:
        """Process length estimates for a single fish.

        Args:
            fish_id: Fish track ID
            data: Dict with 'frame_num' and 'pred_kpts_global_px' lists

        Returns:
            LengthEstimate or None if insufficient data
        """
        pred_kpts_global_px = data["pred_kpts_global_px"]
        frame_nums = data["frame_num"]

        # Check minimum window sizes
        if (
            self.config.length_window_size is not None
            and len(pred_kpts_global_px) < self.config.length_window_size
        ):
            logger.debug(
                "insufficient_length_window",
                fish_id=fish_id,
                num_estimates=len(pred_kpts_global_px),
                required=self.config.length_window_size,
            )
            return None

        if (
            self.config.vel_window_size is not None
            and len(pred_kpts_global_px) < self.config.vel_window_size
        ):
            logger.debug(
                "insufficient_velocity_window",
                fish_id=fish_id,
                num_estimates=len(pred_kpts_global_px),
                required=self.config.vel_window_size,
            )
            return None

        # Apply filters
        filter_result = self.filter.apply_filters(
            pred_kpts_global_px, frame_nums, self.cone_params
        )

        # Determine best estimate
        if filter_result.closest_to_mean_idx is not None:
            frame_id_closest = frame_nums[filter_result.closest_to_mean_idx]
            coords = [
                pred_kpts_global_px[filter_result.closest_to_mean_idx][0],
                pred_kpts_global_px[filter_result.closest_to_mean_idx][1],
            ]
        else:
            frame_id_closest = None
            coords = None

        return LengthEstimate(
            fish_id=fish_id,
            all_lengths_cm=filter_result.mean_length_cm,
            filtered_lengths_cm=filter_result.filtered_mean_length_cm,
            num_filtered=filter_result.num_passed,
            frame_id_closest_to_mean=frame_id_closest,
            coords_px=coords,
        )

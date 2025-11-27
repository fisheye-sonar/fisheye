from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional, List, Union

import structlog

from fisheye.boxes import (
    run_nms,
    normalize_boxes_for_tracking,
    mean_bbox_width_yolo_to_image,
    median_bbox_width_yolo_to_image,
    std_bbox_width_yolo_to_image,
)
from fisheye.common.generic import safe_execution
from fisheye.common.file_system import get_valid_files
from fisheye.configs import YOLODatasetConfig
from fisheye.configs.inference import TrackerConfig, NMSConfig
from fisheye.count.counter import Count
from fisheye.enums import ExportType, UpstreamDirectionTypes
from fisheye.export import save_to_disk, to_mot_txt
from fisheye.format import tracker_output_to_dict_rows, dict_rows_to_mot_format
from fisheye.pipelines import ObjectDetectionPipeline
from fisheye.track.tracker import run_tracker
from fisheye.lengths.length_models import get_model

from fisheye.lengths.measure_utils import get_cone_edges, calc_len
from fisheye.lengths.measure_utils import get_min_edge_distances_pxl
from fisheye.lengths.measure_utils import get_change_in_length
from fisheye.lengths.measure_utils import get_velocity_dev
from fisheye.lengths.measure_video_wise import get_pred_from_video_wise_helper

# MAH 2025-11-24 14:48:40 imports that can probably be removed later when this is tidied out of pipeline
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


logger = structlog.getLogger(__name__)


class DetectTrackCountPipeline:
    """Pipeline for detection, tracking, and counting."""

    def __init__(
        self,
        detect_pipe: Optional[ObjectDetectionPipeline] = None,
        tracker_cfg: Optional[TrackerConfig] = None,
        dataset_cfg: YOLODatasetConfig = None,
    ):
        self.detect_pipe = detect_pipe
        self.tracker_cfg = tracker_cfg if tracker_cfg else TrackerConfig()
        self.nms_config = NMSConfig()
        self.dataset_cfg = dataset_cfg if dataset_cfg else YOLODatasetConfig()

    @safe_execution(default_return=[], max_retries=3, delay=2)
    def _run(
        self,
        file: Union[Path, List[Path]],
        output_dir: Union[str, Path],
        export_types: Optional[List[ExportType]] = None,
        job_id: Optional[str] = None,
        upstream_direction: UpstreamDirectionTypes = UpstreamDirectionTypes.LEFT,
        distance_offset: Union[int, float] = 0.0,
    ) -> List:

        if not output_dir:
            output_dir = file.parent

        logger.info("file_processing_started", file_path=str(file))
        # Shallow copy of YOLODatasetConfig with updated fields
        print(f"MAH put end frame back")
        start_frame = 0
        end_frame = 0
        start_frame = 131
        end_frame = 532
        self.dataset_cfg = replace(
            self.dataset_cfg,
            filepath=file,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self.detect_pipe.load_dataset(self.dataset_cfg)
        metadata = self.detect_pipe.metadata

        if True:
            if self.detect_pipe.apply_nms_batchwise:
                print("Applying NMS batchwise")
                if self.detect_pipe.apply_length_estimates_batchwise:
                    (
                        low_preds,
                        high_preds,
                        low_length_estimates,
                        high_length_estimates,
                    ) = self.detect_pipe()
                else:
                    low_preds, high_preds = self.detect_pipe()
            else:
                print("Applying NMS over all frames")
                detections = self.detect_pipe()

                # Get low confidence for ByteTrack
                self.nms_config.conf = 0.1
                low_output = run_nms(
                    detections.pred_bboxes,  # xyxy format relative to YOLO pixel space
                    metadata.image_meter_width,
                    detections.width,
                    self.dataset_cfg.batch_size,
                    self.nms_config,
                )

                # Get high confidence for ByteTrack
                self.nms_config.conf = 0.3
                high_output = run_nms(
                    detections.pred_bboxes,  # xyxy format relative to YOLO pixel space
                    metadata.image_meter_width,
                    detections.width,
                    self.dataset_cfg.batch_size,
                    self.nms_config,
                )

                # Prepare bounding boxes for tracking pipeline
                low_preds, og_width, og_height = normalize_boxes_for_tracking(
                    detections.image_shape,
                    low_output,  # xyxy format relative to YOLO pixel space
                    detections.width,
                    detections.height,
                    batch_size=self.dataset_cfg.batch_size,
                )
                high_preds, og_width, og_height = normalize_boxes_for_tracking(
                    detections.image_shape,
                    high_output,  # xyxy format relative to YOLO pixel space
                    detections.width,
                    detections.height,
                    batch_size=self.dataset_cfg.batch_size,
                )

            # for k, v in low_preds.items():
            #     if v is not None:
            #         print(f"low_preds {k}: {v.shape}")
            #     else:
            #         print(f"low_preds {k}: {v}")
            # for k, v in high_preds.items():
            #     if v is not None:
            #         print(f"high_preds {k}: {v.shape}")
            #     else:
            #         print(f"high_preds {k}: {v}")
            # for k, v in low_length_estimates.items():
            #     print(f"low_length_estimates {k}: ")
            #     if v is not None:
            #         for v2 in v:
            #             print(f"    {v2}")
            #             print(
            #                 f"\033[91m    {calc_len(v2['pred_kpts_global_px'])*metadata.pixel_meter_size*100:.1f}cm\033[0m"
            #             )
            #     else:
            #         print(f"    {v}")

            tracker_output = run_tracker(
                low_preds,  # xyxy format relative to the original image pixel space
                high_preds,  # xyxy format relative to the original image pixel space
                metadata.image_meter_width,
                metadata.image_meter_height,
                self.tracker_cfg,
            )
            tracker_output_dict = asdict(tracker_output)
            frames_preds = tracker_output_dict["frames"]

        if self.detect_pipe.apply_length_estimates_batchwise:

            tracker_output_dict = tracker_output_to_dict_rows(asdict(tracker_output))

            # add the associated length estimates to the frames_preds
            for frame_pred_idx, frame_pred in enumerate(frames_preds):
                frame_num = frame_pred["frame_num"]
                frame_bs_num = (
                    int(frame_num / self.dataset_cfg.batch_size),
                    frame_num % self.dataset_cfg.batch_size,
                )
                for fish_idx, fish_pred in enumerate(frame_pred["fish"]):
                    fish_id = fish_pred["id"]
                    _bbox = fish_pred["bbox"]
                    _conf = fish_pred["conf"]
                    bbox_index = fish_pred["bbox_index"]
                    det_index = fish_pred["det_index"]
                    if det_index == 0:
                        matched_length_estimate = low_length_estimates[frame_bs_num][
                            bbox_index
                        ]
                    else:
                        print(
                            "\033[41m# MAH 2025-11-26 17:33:08 matched to a high confidence detection, given the way we do our filtering (low first then high) i am not certain this will ever be done, if it is remove this line from the codebase\033[0m"
                        )
                        matched_length_estimate = high_length_estimates[frame_bs_num][
                            bbox_index
                        ]

                    frames_preds[frame_pred_idx]["fish"][fish_idx][
                        "length_estimate"
                    ] = matched_length_estimate

            # print the length estimates
            # for frame_pred in frames_preds:
            #     frame_num = frame_pred["frame_num"]
            #     for fish_pred in frame_pred["fish"]:
            #         fish_id = fish_pred["id"]
            #         length_estimate = fish_pred["length_estimate"][
            #             "pred_kpts_global_px"
            #         ]
            #         print(f"{frame_num=}: {fish_id=}: ")
            #         print(
            #             f"\033[91m    {calc_len(fish_pred['length_estimate']['pred_kpts_global_px'])*metadata.pixel_meter_size*100:.1f}cm\033[0m"
            #         )
            #         print(f"{frame_num=}: {fish_id=}: {length_estimate=}")

            all_length_estimates = {}
            for frame_pred in frames_preds:
                frame_num = frame_pred["frame_num"]
                for fish_pred in frame_pred["fish"]:
                    fish_id = fish_pred["id"]
                    length_estimate = fish_pred["length_estimate"][
                        "pred_kpts_global_px"
                    ]
                    if fish_id not in all_length_estimates:
                        all_length_estimates[fish_id] = {
                            "frame_num": [],
                            "pred_kpts_global_px": [],
                        }
                    all_length_estimates[fish_id]["frame_num"].append(frame_num)
                    all_length_estimates[fish_id]["pred_kpts_global_px"].append(
                        length_estimate
                    )

            (
                _cone_points_left,
                _cone_points_right,
                cone_eq_params_left,
                cone_eq_params_right,
            ) = get_cone_edges(metadata)

            # MAH 2025-11-26 18:40:48 TODO put these in the config
            min_edge_dist_tolerance_px = 10
            vel_delta_tolerance = 15
            length_delta_tolerance_cm = 5
            vel_window_size = 7
            length_window_size = 7

            plot_filters_for_debugging = True

            len_outputs = {}
            for fish_id, data in all_length_estimates.items():
                if (
                    length_window_size is not None
                    and len(data["pred_kpts_global_px"]) < length_window_size
                ):
                    len_outputs[fish_id] = None
                    continue
                if (
                    vel_window_size is not None
                    and len(data["pred_kpts_global_px"]) < vel_window_size
                ):
                    len_outputs[fish_id] = None
                    continue
                pred_kpts_global_px = data["pred_kpts_global_px"]
                frame_nums = data["frame_num"]
                pred_kpts_global_0_cm = [
                    kpts_global_px[0] * metadata.pixel_meter_size * 100
                    for kpts_global_px in pred_kpts_global_px
                ]
                pred_kpts_global_1_cm = [
                    kpts_global_px[1] * metadata.pixel_meter_size * 100
                    for kpts_global_px in pred_kpts_global_px
                ]
                pred_lens_cm = [
                    calc_len(kpts_global_px) * metadata.pixel_meter_size * 100
                    for kpts_global_px in pred_kpts_global_px
                ]
                print(
                    f"id:{fish_id} -- {[int(pred_lens_cm[i]) for i in range(len(pred_lens_cm))]}"
                )

                masks = {}
                masks["all"] = [True] * len(pred_kpts_global_px)

                if min_edge_dist_tolerance_px is not None:
                    min_edge_distances_pxl = []
                    for kpts_global_px in pred_kpts_global_px:
                        edge_dist_i = get_min_edge_distances_pxl(
                            kpts_global_px,
                            cone_eq_params_left[0],
                            cone_eq_params_left[1],
                            cone_eq_params_right[0],
                            cone_eq_params_right[1],
                        )[0]
                        min_edge_distances_pxl.append(edge_dist_i)
                    masks["edge_dist"] = [
                        edge_dist >= min_edge_dist_tolerance_px
                        for edge_dist in min_edge_distances_pxl
                    ]

                if length_delta_tolerance_cm is not None:
                    # filter out the fish with changes in length from a moving average
                    change_in_length, _moving_average_length = get_change_in_length(
                        pred_kpts_global_0_cm,
                        pred_kpts_global_1_cm,
                        frame_nums,
                        window_size=length_window_size,
                        robust=True,
                    )
                    masks["length"] = [
                        abs(change) < length_delta_tolerance_cm
                        for change in change_in_length
                    ]
                    lengths_cm = [
                        calc_len(kpts_global_px) * metadata.pixel_meter_size * 100
                        for kpts_global_px in pred_kpts_global_px
                    ]
                    # check if the frame_nums are sequential
                    if not np.all(np.diff(frame_nums) == 1):
                        print(
                            f"\033[41m # MAH 2025-11-26 18:43:22 frame_nums are not sequential\n{frame_nums=}\033[0m"
                        )
                    if plot_filters_for_debugging:
                        matplotlib.use("TkAgg")

                        fig, ax = plt.subplots(1, 5)
                        for i in range(len(lengths_cm)):
                            c = "green" if masks["length"][i] else "red"
                            ax[0].scatter(frame_nums[i], lengths_cm[i], color=c)
                            ax[1].scatter(frame_nums[i], change_in_length[i], color=c)
                            kpts = pred_kpts_global_px[i]
                            ax[2].plot(
                                [kpts[0][0], kpts[1][0]],
                                [kpts[0][1], kpts[1][1]],
                                color=c,
                            )
                        ax[0].plot(frame_nums, _moving_average_length)
                        ax[1].axhline(y=length_delta_tolerance_cm)
                        ax[1].axhline(y=-length_delta_tolerance_cm)
                        ax[0].set_title("Lengths")
                        ax[1].set_title("Change in length")

                if vel_delta_tolerance is not None:
                    # filter out the fish with deviations
                    velocity_deviations = get_velocity_dev(
                        pred_kpts_global_0_cm,
                        pred_kpts_global_1_cm,
                        frame_nums,
                        window_size=vel_window_size,
                    )  # MAH 2025-11-24 12:07:56 TODO this needs to take in the fact that they may not be sequential frames
                    masks["velocity"] = [
                        vel < vel_delta_tolerance for vel in velocity_deviations
                    ]
                    if plot_filters_for_debugging:
                        for i in range(len(velocity_deviations)):
                            if masks["length"][i]:
                                if masks["velocity"][i]:
                                    c = "green"
                                else:
                                    c = "red"
                            else:
                                if masks["velocity"][i]:
                                    c = "orange"
                                else:
                                    c = "blue"

                            ax[3].scatter(
                                frame_nums[i], velocity_deviations[i], color=c
                            )
                            kpts = pred_kpts_global_px[i]
                            ax[4].plot(
                                [kpts[0][0], kpts[1][0]],
                                [kpts[0][1], kpts[1][1]],
                                color=c,
                            )
                        ax[3].set_title("Velocity deviations")
                if plot_filters_for_debugging:
                    ax[2].set_ylim(ax[2].get_ylim()[1], ax[2].get_ylim()[0])
                    ax[4].set_ylim(ax[4].get_ylim()[1], ax[4].get_ylim()[0])
                    fig.suptitle(f"Length exclusions for fish {fish_id}")
                    plt.show()

                # apply the filters and get the masks
                combined_filter_mask = np.all(
                    np.array([v for v in masks.values()]) == 1, axis=0
                ).astype(int)

                num_filtered = sum(combined_filter_mask)
                if num_filtered == 0:
                    filtered_len_cm = np.nan
                else:
                    filtered_len_cm = np.mean(
                        [
                            pred_lens_cm[i]
                            for i in range(len(pred_lens_cm))
                            if combined_filter_mask[i]
                        ]
                    )

                # get the index of the closest to the mean
                clostest_to_mean_pred_len_indx = None
                if np.sum(combined_filter_mask):
                    pred_lens_filtered = pred_lens_cm * combined_filter_mask
                    # mean where not zero
                    pred_lens_filtered_mean = np.mean(
                        pred_lens_filtered[pred_lens_filtered != 0]
                    )
                    clostest_to_mean_pred_len_ind = np.argmin(
                        abs(pred_lens_filtered - pred_lens_filtered_mean)
                    )

                    if combined_filter_mask[clostest_to_mean_pred_len_ind]:
                        clostest_to_mean_pred_len_indx = clostest_to_mean_pred_len_ind

                if clostest_to_mean_pred_len_indx is not None:
                    frame_id_closest_to_mean = frame_nums[
                        clostest_to_mean_pred_len_indx
                    ]

                    coords = [
                        pred_kpts_global_px[clostest_to_mean_pred_len_indx][0],
                        pred_kpts_global_px[clostest_to_mean_pred_len_indx][1],
                    ]
                else:
                    frame_id_closest_to_mean = None

                    coords = None

                len_outputs[fish_id] = {
                    "fish_id": fish_id,
                    "all_lengths_cm": np.mean(pred_lens_cm),
                    "filtered_lengths_cm": filtered_len_cm,
                    "num_filtered": num_filtered,
                    "frame_id_closest_to_mean": frame_id_closest_to_mean,
                    "coords_px": coords,
                }

                print(f"{fish_id}: {len_outputs[fish_id]=}")
        else:
            # this is significantly less efficient than the batchwise approach
            additional_bbox_padding_px = 5
            len_outputs = get_pred_from_video_wise_helper(
                frames_preds,
                metadata,
                vel_window_size,
                length_window_size,
                vel_delta_tolerance,
                length_delta_tolerance_cm,
                min_edge_dist_tolerance_px,
                self.detect_pipe.dataset,
                start_frame,
                additional_bbox_padding_px,
            )

        formatted_yolo_tracks = tracker_output_to_dict_rows(asdict(tracker_output))

        if export_types is None:
            export_types_list = []

        elif isinstance(export_types, ExportType):
            export_types_list = [export_types]

        else:
            export_types_list = export_types

        if ExportType.MOT in export_types_list:
            mot_tracks = dict_rows_to_mot_format(
                formatted_yolo_tracks, metadata.xdim, metadata.ydim
            )
            to_mot_txt(mot_tracks, output_dir, file.stem)

        (left_count, right_count), crossing_frames = Count().count(
            formatted_yolo_tracks
        )

        if crossing_frames and (left_count or right_count):
            # Using the same naming conventions in ARISFish Software
            formatted_crossings = [
                {
                    "Source.Name": Path(file).name,
                    "Frame#": frame,
                    "Dir": "Up" if upstream_direction == "left" else "Down",
                    "ID": track,
                    "bbox": bbox,  # [x_center, y_center, width, height] relative to original image space
                    "metadata": metadata,
                }
                for track, frame, bbox in crossing_frames["left"]
            ] + [
                {
                    "Source.Name": Path(file).name,
                    "Frame#": frame,
                    "Dir": "Up" if upstream_direction == "right" else "Down",
                    "ID": track,
                    "bbox": bbox,  # [x_center, y_center, width, height] relative to original image space
                    "metadata": metadata,
                }
                for track, frame, bbox in crossing_frames["right"]
            ]

            # TODO (MHV): Try/except is temporary until this logic has been tested more vigorously
            try:
                formated_yolo_tracks = None
                # Extract unique track IDs
                unique_ids = {row["id"] for row in formatted_yolo_tracks}
                num_tracks = len(unique_ids)

                # Extract all bboxes from the formatted crossings
                all_bboxes = [f["bbox"] for f in formatted_crossings]
                std_bbox_width = std_bbox_width_yolo_to_image(
                    all_bboxes, metadata.image_meter_width
                )
                avg_bbox_width = mean_bbox_width_yolo_to_image(
                    all_bboxes, metadata.image_meter_width
                )
                median_bbox_width = median_bbox_width_yolo_to_image(
                    all_bboxes, metadata.image_meter_width
                )

                # Log stats for current ARIS file
                logger.info(
                    "processed_file_stats",
                    num_counts=len(formatted_crossings),
                    num_tracks=num_tracks,
                    avg_bbox_width_meters=avg_bbox_width,
                    median_bbox_width_meters=median_bbox_width,
                    std_bbox_width_meters=std_bbox_width,
                )

            except Exception:
                # Silently skip stats if calculation fails.
                pass

        else:
            formatted_crossings = [
                {
                    "Source.Name": Path(file).name,
                    "Frame#": None,
                    "Dir": None,
                    "ID": None,
                    "bbox": None,
                    "metadata": metadata,
                }
            ]

            logger.warning("no_counts", file_path=str(file))

        print(f"{formatted_crossings=}")
        exit()
        remaining_export_types = [
            et
            for et in export_types_list
            if et != ExportType.MOT and (et != ExportType.SUMMARY_CSV)
        ]
        save_to_disk(
            [formatted_crossings],
            output_dir,
            export_types=remaining_export_types,
            job_id=job_id,
            distance_offset=distance_offset,
        )

        return formatted_crossings

    def run(
        self,
        file: Union[str, Path, List[Union[str, Path]]],
        output_dir: Union[str, Path],
        export_types: Optional[List[ExportType]] = None,
        job_id: Optional[str] = None,
        upstream_direction: UpstreamDirectionTypes = UpstreamDirectionTypes.LEFT,
        distance_offset: Union[int, float] = 0.0,
    ) -> Union[List[List[dict]], List[dict]]:
        """Run preprocessing, detection, tracking, and counting on frames.

        Args:
            file (List[str] | str): File(s) to process. Must be a path to an ARIS file or a directory holding ARIS files
            output_dir (str): Output directory to save results to
            export_types (Optional[List[ExportType]]): List of ExportType objects to export to
            job_id (Optional[str]): Job ID
            upstream_direction (Optional[UpstreamDirectionTypes]): Upstream direction

        Returns:
            dict: Tracking results and counts.
        """

        valid_files = get_valid_files(file, output_dir)
        if not valid_files:
            logger.error(
                f"Unable to process valid files. Please verify that the file path is correct and that the "
                f"file hasn't already been processed."
            )
            return []

        results = [
            self._run(
                f, output_dir, export_types, job_id, upstream_direction, distance_offset
            )
            for f in valid_files
        ]

        return results

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
from fisheye.lengths.measure import get_pred_from_dir
from fisheye.lengths.measure_utils import get_cone_edges, calc_len

# MAH 2025-11-24 14:48:40 imports that can probably be removed later when this is tidied out of pipeline
import torch
from math import floor, ceil
import numpy as np

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
        start_frame = 315
        end_frame = 415
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
                    low_preds, high_preds, length_estimates = self.detect_pipe()
                else:
                    low_preds, high_preds = self.detect_pipe()
            else:
                print("Applying NMS over all frames")

                detections = self.detect_pipe()

                # detections = self.detect_pipe._forward(nms_config=self.nms_config)
                print(f"{len(detections.pred_bboxes[0])=}")
                print(f"{detections.pred_bboxes[0].shape=}")
                print(f"{len(detections.pred_bboxes[1])=}")

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
                print(f"{high_output=}")

                print(f"{len(low_output)=}")
                print(f"{len(high_output)=}")
                print(f"{len(high_output[0])=}")
                print(f"{len(high_output[0][0])=}")
                print(f"{high_output[0][0]=}")
                for i, high_b in enumerate(high_output):
                    for ii, hb in enumerate(high_b):
                        print(f"{i}, {ii}: {hb=}")

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

            # print(f"{low_preds=}")
            # print(f"{high_preds=}")

            # print(f"{len(low_preds)=}")
            print(f"{len(low_preds)=}")
            print(f"{len(high_preds)=}")
            print(f"{len(list(high_preds.keys()))=}")
            for k, v in low_preds.items():
                if v is not None:
                    print(f"low_preds {k}: {v.shape}")
                else:
                    print(f"low_preds {k}: {v}")
            for k, v in length_estimates.items():
                print(f"length_estimates {k}: ")
                if v is not None:
                    for v2 in v:
                        print(f"    {v2}")
                        print(
                            f"\033[91m    {calc_len(v2['pred_kpts'])*metadata.pixel_meter_size*100:.1f}cm\033[0m"
                        )
                else:
                    print(f"    {v}")

            tracker_output = run_tracker(
                low_preds,  # xyxy format relative to the original image pixel space
                high_preds,  # xyxy format relative to the original image pixel space
                metadata.image_meter_width,
                metadata.image_meter_height,
                self.tracker_cfg,
            )
            tracker_output_dict = asdict(tracker_output)
            frames_preds = tracker_output_dict["frames"]
            print(f"{frames_preds=}")
        if self.detect_pipe.apply_length_estimates_batchwise:

            length_estimates = length_estimates
            tracker_output_dict = tracker_output_to_dict_rows(asdict(tracker_output))
            print(f"{tracker_output_dict=}")

            print(
                "we need to pull the lengths that are associated with the tracker output bboxes , and then filter them based on the length estimates, edge etc"
            )
            exit()
            (
                _cone_points_left,
                _cone_points_right,
                cone_eq_params_left,
                cone_eq_params_right,
            ) = get_cone_edges(metadata)

            for frame_num, pred_len_output in length_estimates.items():
                for pred_info in pred_len_output["pred_infos"]:
                    _ = get_min_edge_distances_pxl(
                        pred_info["pred_kpts_global_px"],
                        cone_eq_params_left,
                        cone_eq_params_right,
                    )[0]

            for fish_id in fish_ids:
                masks = {}
                masks["all"] = [True] * len(pred_lens_cm[fish_id])
                if min_edge_dist_tolerance is not None:
                    masks["edge_dist"] = [
                        edge_dist >= min_edge_dist_tolerance
                        for edge_dist in min_edge_distances_pxl[fish_id]
                    ]
                if vel_delta_tolerance is not None:
                    # filter out the fish with deviations
                    velocity_deviations = get_velocity_dev(
                        pred_kpts_global_0_cm[fish_id],
                        pred_kpts_global_1_cm[fish_id],
                        window_size=vel_window_size,
                    )  # MAH 2025-11-24 12:07:56 TODO this needs to take in the fact that they may not be sequential frames
                    masks["velocity"] = [
                        vel < vel_delta_tolerance for vel in velocity_deviations
                    ]

                if length_delta_tolerance is not None:
                    # filter out the fish with changes in length from a moving average
                    change_in_length, _moving_average_length = get_change_in_length(
                        pred_kpts_global_0_cm[fish_id],
                        pred_kpts_global_1_cm[fish_id],
                        window_size=length_window_size,
                        robust=True,
                    )
                    masks["length"] = [
                        abs(change) < length_delta_tolerance
                        for change in change_in_length
                    ]

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
                            pred_lens_cm[fish_id][i]
                            for i in range(len(pred_lens_cm[fish_id]))
                            if combined_filter_mask[i]
                        ]
                    )

                # get the index of the closest to the mean
                if np.sum(combined_filter_mask):
                    pred_lens_filtered = pred_lens_cm[fish_id] * combined_filter_mask
                    # mean where not zero
                    pred_lens_filtered_mean = np.mean(
                        pred_lens_filtered[pred_lens_filtered != 0]
                    )
                    clostest_to_mean_pred_len_indx = np.argmin(
                        abs(pred_lens_filtered - pred_lens_filtered_mean)
                    )
                    if not combined_filter_mask[clostest_to_mean_pred_len_indx]:
                        frame_id_closest_to_mean = None
                    else:
                        frame_id_closest_to_mean = frame_nums[fish_id][
                            clostest_to_mean_pred_len_indx
                        ]
                else:
                    frame_id_closest_to_mean = None
                    pred_lens_filtered_mean = None

                if clostest_to_mean_pred_len_indx is not None:
                    coords = [
                        pred_kpts_global_0_px[fish_id][clostest_to_mean_pred_len_indx],
                        pred_kpts_global_1_px[fish_id][clostest_to_mean_pred_len_indx],
                    ]
                else:
                    coords = None

                len_outputs[fish_id] = {
                    "pred_length_cm": pred_lens_filtered_mean,
                    "filtered_lengths_cm": filtered_len_cm,
                    "num_filtered": num_filtered,
                    "frame_id_closest_to_mean": frame_id_closest_to_mean,
                    "coords": coords,
                }

                print(f"{len_outputs[fish_id]=}")
            return len_outputs
        else:
            if False:
                model_type = "unet"
                unet_double_conv = False
                model_input_channels = 1
                model_input_channels = 3
                load_model_path = "/home/mahobley/Code/fisheye-dev/head_tail/checkpoints/crop_after_model/model_150.pth"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                crop_after_model = True
                additional_bbox_padding_px = 5

                vel_window_size = 7
                length_window_size = 7
                vel_delta_tolerance = None
                length_delta_tolerance = None
                min_edge_dist_tolerance = 10
                min_edge_dist_tolerance = None

                mapTokpt_differentiable = False
                mapTokpt_round_to_integer = False

                crop_info = []
                for frame_pred in frames_preds:
                    fn = frame_pred["frame_num"]
                    frame_crop_infos = []
                    for fish_pred in frame_pred["fish"]:
                        fish_id = fish_pred["id"]

                        pred_bbox = fish_pred["bbox"]
                        pred_bbox_xyxy = [
                            floor(pred_bbox[0] * metadata.xdim),
                            floor(pred_bbox[1] * metadata.ydim),
                            ceil(pred_bbox[2] * metadata.xdim),
                            ceil(pred_bbox[3] * metadata.ydim),
                        ]

                        right_space = metadata.xdim - pred_bbox_xyxy[2]
                        bottom_space = metadata.ydim - pred_bbox_xyxy[3]

                        crop_l = max(0, pred_bbox_xyxy[0] - additional_bbox_padding_px)
                        crop_t = max(0, pred_bbox_xyxy[1] - additional_bbox_padding_px)
                        crop_r = max(1, right_space - additional_bbox_padding_px)
                        crop_b = max(1, bottom_space - additional_bbox_padding_px)

                        frame_crop_infos.append(
                            {
                                "fish_id": fish_id,
                                "crop_l": crop_l,
                                "crop_t": crop_t,
                                "crop_r": crop_r,
                                "crop_b": crop_b,
                            }
                        )
                    crop_info.append(
                        {
                            "frame_num": fn,
                            "frame_crop_infos": frame_crop_infos,
                        }
                    )

                model = get_model(
                    model_type,
                    model_input_channels,
                    unet_double_conv,
                    load_model_path,
                    device,
                )

                (
                    _cone_points_left,
                    _cone_points_right,
                    cone_eq_params_left,
                    cone_eq_params_right,
                ) = get_cone_edges(metadata)

                dataset = self.detect_pipe.dataset

                pred_len_outputs = get_pred_from_dir(
                    crop_info,
                    dataset,
                    model,
                    crop_after_model,
                    pxl_to_cm_scale=metadata.pixel_meter_size * 100,
                    vel_window_size=vel_window_size,
                    length_window_size=length_window_size,
                    device=device,
                    cone_eq_params_left=cone_eq_params_left,
                    cone_eq_params_right=cone_eq_params_right,
                    vel_delta_tolerance=vel_delta_tolerance,
                    length_delta_tolerance=length_delta_tolerance,
                    min_edge_dist_tolerance=min_edge_dist_tolerance,
                    model_input_channels=model_input_channels,
                    mapTokpt_differentiable=mapTokpt_differentiable,
                    mapTokpt_round_to_integer=mapTokpt_round_to_integer,
                )

                # add start_frame to frame_id_closest_to_mean
                for fish_id, pred_len_output in pred_len_outputs.items():
                    pred_len_output["frame_id_closest_to_mean"] += start_frame

                print(f"{pred_len_outputs=}")
                exit()

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

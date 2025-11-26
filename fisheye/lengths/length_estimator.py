import torch
from math import floor, ceil
from fisheye.lengths.length_models import get_model
from fisheye.lengths.measure_utils import get_cone_edges
from fisheye.lengths.measure import (
    mapTokpt,
    average_brightness_on_line,
    calc_len,
    get_min_edge_distances_pxl,
)
from fisheye.lengths.measure_utils import get_velocity_dev, get_change_in_length
from tqdm import tqdm
import numpy as np


class LengthEstimator:
    def __init__(self, metadata):
        self.metadata = metadata

        self.xdim = metadata.xdim
        self.ydim = metadata.ydim

        # MAH 2025-11-25 16:32:57 TODO: for testing purposes, this should be removed as its expecting to be run on the orginal image size, need to figure ouyt dataloader returning both resized and not
        # self.ydim = 960
        # self.xdim = 512

        model_type = "unet"
        unet_double_conv = False
        self.model_input_channels = 1
        self.model_input_channels = 3
        load_model_path = "/home/mahobley/Code/fisheye-dev/head_tail/checkpoints/crop_after_model/model_150.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.crop_after_model = True
        self.additional_bbox_padding_px = 25

        self.vel_window_size = 7
        self.length_window_size = 7
        self.vel_delta_tolerance = None
        self.length_delta_tolerance = None
        self.min_edge_dist_tolerance = 10
        self.min_edge_dist_tolerance = None

        self.mapTokpt_differentiable = False
        self.mapTokpt_round_to_integer = False

        self.model = get_model(
            model_type,
            self.model_input_channels,
            unet_double_conv,
            load_model_path,
            device,
        )

        self.pxl_to_cm_scale = self.metadata.pixel_meter_size * 100

        (
            _cone_points_left,
            _cone_points_right,
            self.cone_eq_params_left,
            self.cone_eq_params_right,
        ) = get_cone_edges(self.metadata)

        self.model.eval()

    def get_pred_from_img(
        self,
        img,
        crops_l=None,
        crops_t=None,
        crops_r=None,
        crops_b=None,
    ):
        crop = (
            crops_l is not None
            and crops_t is not None
            and crops_r is not None
            and crops_b is not None
        )

        if crop:
            if not isinstance(crops_l, list):
                crops_l = [crops_l]
            if not isinstance(crops_t, list):
                crops_t = [crops_t]
            if not isinstance(crops_r, list):
                crops_r = [crops_r]
            if not isinstance(crops_b, list):
                crops_b = [crops_b]
        else:
            crops_l = [0]
            crops_t = [0]
            crops_r = [1]
            crops_b = [1]

        if self.model_input_channels == 1:
            if img.shape[1] == 3:
                # if given a 3 channel image (prev, current, next) we want to use the current channel
                img = img[:, 1:2, :, :]
        self.model.eval()
        if True:
            # whole image
            pred = self.model(img.float())
        else:
            # initial crop to have a smaller input image
            padd_for_receptive_field = 100
            min_crop_l = max(0, min(crops_l) - padd_for_receptive_field)
            min_crop_t = max(0, min(crops_t) - padd_for_receptive_field)
            min_crop_r = max(1, min(crops_r) - padd_for_receptive_field)
            min_crop_b = max(1, min(crops_b) - padd_for_receptive_field)

            pred = torch.zeros(img.shape[0], 2, img.shape[2], img.shape[3])
            img_init_crop = img[:, :, min_crop_t:-min_crop_b, min_crop_l:-min_crop_r]
            print(f"{img.shape=}")
            print(f"{img_init_crop.shape=}")
            pred_init_crop = model(img_init_crop)
            print(f"{pred_init_crop.shape=}")
            # uncrop the pred
            pred[:, :, min_crop_t:-min_crop_b, min_crop_l:-min_crop_r] = pred_init_crop

        outputs = []
        for crop_l, crop_t, crop_r, crop_b in zip(crops_l, crops_t, crops_r, crops_b):
            pred_cropped = pred[:, :, crop_t:-crop_b, crop_l:-crop_r]
            img_cropped = img[:, :, crop_t:-crop_b, crop_l:-crop_r]
            print(f"{crop_t=}")
            print(f"{crop_b=}")
            print(f"{crop_l=}")
            print(f"{crop_r=}")
            print(f"{img.shape=}")
            print(f"{img_cropped.shape=}")
            print(f"{pred_cropped.shape=}")

            pred_kpts = mapTokpt(
                pred_cropped,
                differentiable=self.mapTokpt_differentiable,
                round_to_integer=self.mapTokpt_round_to_integer,
            )[0]

            print(f"{pred_kpts.shape=}")
            print(f"{pred_kpts=}")
            print(f"{calc_len(pred_kpts)=}")
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(pred_cropped[0, 0].cpu().numpy())
            # ax[1].imshow(pred_cropped[0, 1].cpu().numpy())
            # ax[2].imshow(vis_3_channel_img(img_cropped[0]).cpu().numpy())
            # ax[2].scatter(
            #     pred_kpts[:, 0].cpu().numpy(),
            #     pred_kpts[:, 1].cpu().numpy(),
            #     color="green",
            #     marker="x",
            # )
            # plt.savefig(f"pred_kpts_{crop_l}_{crop_t}_{crop_r}_{crop_b}.png")
            # plt.show()

            pred_kpts_global = pred_kpts.cpu().numpy().copy()
            pred_kpts_global[:, 0] += crop_l
            pred_kpts_global[:, 1] += crop_t

            output = {
                "pred_kpts": pred_kpts.cpu().numpy(),
                "pred_kpts_global_px": pred_kpts_global,
            }
            if False:
                peak_heatmap_brightness_0 = torch.max(pred_cropped[:, 0]).item()
                peak_heatmap_brightness_1 = torch.max(pred_cropped[:, 1]).item()
                output["peak_heatmap_brightness_0"] = peak_heatmap_brightness_0
                output["peak_heatmap_brightness_1"] = peak_heatmap_brightness_1
                average_brightness_head_to_tail = average_brightness_on_line(
                    img_cropped[0, 1].cpu().numpy(),
                    pred_kpts[0].cpu().numpy(),
                    pred_kpts[1].cpu().numpy(),
                    method="bresenham",
                )
                output["average_brightness_head_to_tail"] = (
                    average_brightness_head_to_tail
                )
            outputs.append(output)
        return outputs

    def get_pred_from_batch(
        self,
        crop_info,
        frames_batch,
        # device,
    ):

        # ml, bl = self.cone_eq_params_left
        # mr, br = self.cone_eq_params_right

        # all_fish_ids = set(
        #     [
        #         fish["fish_id"]
        #         for frame in crop_info
        #         for fish in frame["frame_crop_infos"]
        #     ]
        # )

        # frame_nums = {fish_id: [] for fish_id in all_fish_ids}
        # pred_lens_cm = {fish_id: [] for fish_id in all_fish_ids}
        # pred_kpts_global_0_px = {fish_id: [] for fish_id in all_fish_ids}
        # pred_kpts_global_1_px = {fish_id: [] for fish_id in all_fish_ids}
        # pred_kpts_global_0_cm = {fish_id: [] for fish_id in all_fish_ids}
        # pred_kpts_global_1_cm = {fish_id: [] for fish_id in all_fish_ids}
        # min_edge_distances_pxl = {fish_id: [] for fish_id in all_fish_ids}
        # av_brightnesses = {fish_id: [] for fish_id in all_fish_ids}
        # peak_heatmap_brightnesses_0 = {fish_id: [] for fish_id in all_fish_ids}
        # peak_heatmap_brightnesses_1 = {fish_id: [] for fish_id in all_fish_ids}
        # pred_kpts = {fish_id: [] for fish_id in all_fish_ids}
        # len_outputs = {}

        len_outputs = {}

        with torch.no_grad():
            for frame_crop_info in tqdm(crop_info):
                frame_num = frame_crop_info["frame_num"]

                print(f"{frame_num=} {frame_crop_info=}")
                if frame_crop_info["frame_crop_infos"] == []:
                    continue
                crop_ls = [
                    fish_crop_info["crop_l"]
                    for fish_crop_info in frame_crop_info["frame_crop_infos"]
                ]
                crop_ts = [
                    fish_crop_info["crop_t"]
                    for fish_crop_info in frame_crop_info["frame_crop_infos"]
                ]
                crop_rs = [
                    fish_crop_info["crop_r"]
                    for fish_crop_info in frame_crop_info["frame_crop_infos"]
                ]
                crop_bs = [
                    fish_crop_info["crop_b"]
                    for fish_crop_info in frame_crop_info["frame_crop_infos"]
                ]
                if frame_num == 0:
                    # MAH 2025-11-24 14:26:35 this isnt ideal as not how its trained, should probably just skip the first and last frames
                    frames = torch.stack(
                        [
                            frames_batch[min(len(frames_batch) - 1, frame_num + 2)],
                            frames_batch[frame_num],
                            frames_batch[min(len(frames_batch) - 1, frame_num + 1)],
                        ],
                        dim=0,
                    )
                elif frame_num == len(frames_batch) - 1:
                    # MAH 2025-11-24 14:26:35 this isnt ideal as not how its trained, should probably just skip the first and last frames
                    frames = torch.stack(
                        [
                            frames_batch[max(0, frame_num - 1)],
                            frames_batch[frame_num],
                            frames_batch[max(0, frame_num - 2)],
                        ],
                        dim=0,
                    )
                else:
                    frames = frames_batch[frame_num - 1 : frame_num + 2]

                # frame_images_previous_bgs = torch.from_numpy(
                #     frame_images_previous[0][:, :, 1]
                # ).float()
                # frame_images_current_bgs = torch.from_numpy(
                #     frame_images_current[0][:, :, 1]
                # ).float()
                # frame_images_next_bgs = torch.from_numpy(
                #     frame_images_next[0][:, :, 1]
                # ).float()

                print(f"{frames.shape=}")
                print(f"taking the middle (bgs) channel")
                frames_bgs = frames[:, 1]
                print(f"{frames_bgs.shape=}")

                frames_bgs -= 255 / 2
                frames_bgs[frames_bgs < 0] = 0
                frames_bgs /= torch.max(frames_bgs)
                frames_bgs = frames_bgs.unsqueeze(0)
                # frames_bgs = frames_bgs.to(device)
                print(f"{frames_bgs.shape=}")
                pred_infos = self.get_pred_from_img(
                    frames_bgs,
                    crops_l=crop_ls,
                    crops_t=crop_ts,
                    crops_r=crop_rs,
                    crops_b=crop_bs,
                )
                print(f"{pred_infos=}")
                len_outputs[frame_num] = pred_infos

        #         for fish_id, pred_info in zip(fish_ids, pred_infos):
        #             frame_nums[fish_id].append(frame_num)
        #             pred_lens_cm[fish_id].append(
        #                 calc_len(pred_info["pred_kpts"]) * self.pxl_to_cm_scale
        #             )
        #             pred_kpts_global_0_px[fish_id].append(
        #                 pred_info["pred_kpts_global_px"][0]
        #             )
        #             pred_kpts_global_1_px[fish_id].append(
        #                 pred_info["pred_kpts_global_px"][1]
        #             )
        #             pred_kpts_global_0_cm[fish_id].append(
        #                 pred_info["pred_kpts_global_px"][0] * self.pxl_to_cm_scale
        #             )
        #             pred_kpts_global_1_cm[fish_id].append(
        #                 pred_info["pred_kpts_global_px"][1] * self.pxl_to_cm_scale
        #             )
        #             min_edge_distances_pxl[fish_id].append(
        #                 get_min_edge_distances_pxl(
        #                     pred_info["pred_kpts_global_px"], ml, bl, mr, br
        #                 )[0]
        #             )
        #             av_brightnesses[fish_id].append(
        #                 pred_info["average_brightness_head_to_tail"]
        #             )
        #             peak_heatmap_brightnesses_0[fish_id].append(
        #                 pred_info["peak_heatmap_brightness_0"]
        #             )
        #             peak_heatmap_brightnesses_1[fish_id].append(
        #                 pred_info["peak_heatmap_brightness_1"]
        #             )

        # for fish_id in fish_ids:
        #     masks = {}
        #     masks["all"] = [True] * len(pred_lens_cm[fish_id])
        #     if self.min_edge_dist_tolerance is not None:
        #         masks["edge_dist"] = [
        #             edge_dist >= self.min_edge_dist_tolerance
        #             for edge_dist in min_edge_distances_pxl[fish_id]
        #         ]
        #     if self.vel_delta_tolerance is not None:
        #         # filter out the fish with deviations
        #         velocity_deviations = get_velocity_dev(
        #             pred_kpts_global_0_cm[fish_id],
        #             pred_kpts_global_1_cm[fish_id],
        #             window_size=self.vel_window_size,
        #         )  # MAH 2025-11-24 12:07:56 TODO this needs to take in the fact that they may not be sequential frames
        #         masks["velocity"] = [
        #             vel < self.vel_delta_tolerance for vel in velocity_deviations
        #         ]

        #     if self.length_delta_tolerance is not None:
        #         # filter out the fish with changes in length from a moving average
        #         change_in_length, _moving_average_length = get_change_in_length(
        #             pred_kpts_global_0_cm[fish_id],
        #             pred_kpts_global_1_cm[fish_id],
        #             window_size=self.length_window_size,
        #             robust=True,
        #         )
        #         masks["length"] = [
        #             abs(change) < self.length_delta_tolerance
        #             for change in change_in_length
        #         ]

        #     # apply the filters and get the masks
        #     combined_filter_mask = np.all(
        #         np.array([v for v in masks.values()]) == 1, axis=0
        #     ).astype(int)

        #     num_filtered = sum(combined_filter_mask)
        #     if num_filtered == 0:
        #         filtered_len_cm = np.nan
        #     else:
        #         filtered_len_cm = np.mean(
        #             [
        #                 pred_lens_cm[fish_id][i]
        #                 for i in range(len(pred_lens_cm[fish_id]))
        #                 if combined_filter_mask[i]
        #             ]
        #         )

        #     # get the index of the closest to the mean
        #     if np.sum(combined_filter_mask):
        #         pred_lens_filtered = pred_lens_cm[fish_id] * combined_filter_mask
        #         # mean where not zero
        #         pred_lens_filtered_mean = np.mean(
        #             pred_lens_filtered[pred_lens_filtered != 0]
        #         )
        #         clostest_to_mean_pred_len_indx = np.argmin(
        #             abs(pred_lens_filtered - pred_lens_filtered_mean)
        #         )
        #         if not combined_filter_mask[clostest_to_mean_pred_len_indx]:
        #             frame_id_closest_to_mean = None
        #         else:
        #             frame_id_closest_to_mean = frame_nums[fish_id][
        #                 clostest_to_mean_pred_len_indx
        #             ]
        #     else:
        #         frame_id_closest_to_mean = None
        #         pred_lens_filtered_mean = None

        #     if clostest_to_mean_pred_len_indx is not None:
        #         coords = [
        #             pred_kpts_global_0_px[fish_id][clostest_to_mean_pred_len_indx],
        #             pred_kpts_global_1_px[fish_id][clostest_to_mean_pred_len_indx],
        #         ]
        #     else:
        #         coords = None

        #     len_outputs[fish_id] = {
        #         "pred_length_cm": pred_lens_filtered_mean,
        #         "filtered_lengths_cm": filtered_len_cm,
        #         "num_filtered": num_filtered,
        #         "frame_id_closest_to_mean": frame_id_closest_to_mean,
        #         "coords": coords,
        #     }

        #     print(f"{len_outputs[fish_id]=}")
        return len_outputs

    def run(self, frames_batch, pred_bboxes):
        return self.get_length_estimates(frames_batch, pred_bboxes)

    def get_length_estimates(self, frames_batch, pred_bboxes_batch):
        print(f"{frames_batch.shape=}")
        print(f"{len(pred_bboxes_batch)=}")
        crop_info = []
        for batch_idx_frame_idx, frame_pred_bboxes in pred_bboxes_batch.items():
            frame_crop_infos = []
            frame_idx = batch_idx_frame_idx[1]

            if frame_pred_bboxes is None:
                continue

            print(f"{frame_idx=}")
            print(f"{frame_pred_bboxes=}")
            print(f"{frame_pred_bboxes.shape=}")
            for pred_bbox in frame_pred_bboxes:

                pred_bbox_xyxy = [
                    floor(pred_bbox[0] * self.xdim),
                    floor(pred_bbox[1] * self.ydim),
                    ceil(pred_bbox[2] * self.xdim),
                    ceil(pred_bbox[3] * self.ydim),
                ]

                right_space = self.xdim - pred_bbox_xyxy[2]
                bottom_space = self.ydim - pred_bbox_xyxy[3]

                crop_l = max(0, pred_bbox_xyxy[0] - self.additional_bbox_padding_px)
                crop_t = max(0, pred_bbox_xyxy[1] - self.additional_bbox_padding_px)
                crop_r = max(1, right_space - self.additional_bbox_padding_px)
                crop_b = max(1, bottom_space - self.additional_bbox_padding_px)

                frame_crop_infos.append(
                    {
                        "crop_l": crop_l,
                        "crop_t": crop_t,
                        "crop_r": crop_r,
                        "crop_b": crop_b,
                    }
                )
            crop_info.append(
                {
                    "frame_num": frame_idx,
                    "frame_crop_infos": frame_crop_infos,
                }
            )
        print(f"{crop_info=}")

        pred_len_outputs = self.get_pred_from_batch(
            crop_info,
            frames_batch,
        )

        # add start_frame to frame_id_closest_to_mean
        # for fish_id, pred_len_output in pred_len_outputs.items():
        #     pred_len_output["frame_id_closest_to_mean"] += start_frame
        for frame_num, pred_len_output in pred_len_outputs.items():
            print(f"\033[91m")
            print(f"{frame_num=}:\n\033[93m")
            for v in pred_len_output:
                for k, v2 in v.items():
                    print(f"    {k=}: {v2=}")
        print(f"\033[0m")
        return pred_len_outputs

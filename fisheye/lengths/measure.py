from fisheye.lengths.measure_utils import (
    mapTokpt,
    average_brightness_on_line,
    get_velocity_dev,
    get_change_in_length,
    get_min_edge_distances_pxl,
    calc_len,
    vis_3_channel_img,
)
import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np  # MAH 2025-11-24 15:14:45 remove this later if possible
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
import matplotlib


def get_pred_from_img(
    img,
    model,
    crops_l=None,
    crops_t=None,
    crops_r=None,
    crops_b=None,
    model_input_channels=1,
    mapTokpt_differentiable=False,
    mapTokpt_round_to_integer=False,
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

    if model_input_channels == 1:
        if img.shape[1] == 3:
            # if given a 3 channel image (prev, current, next) we want to use the current channel
            img = img[:, 1:2, :, :]

    if True:
        # whole image
        pred = model(img)
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

        peak_heatmap_brightness_0 = torch.max(pred_cropped[:, 0]).item()
        peak_heatmap_brightness_1 = torch.max(pred_cropped[:, 1]).item()

        pred_kpts = mapTokpt(
            pred_cropped,
            differentiable=mapTokpt_differentiable,
            round_to_integer=mapTokpt_round_to_integer,
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

        average_brightness_head_to_tail = average_brightness_on_line(
            img_cropped[0, 1].cpu().numpy(),
            pred_kpts[0].cpu().numpy(),
            pred_kpts[1].cpu().numpy(),
            method="bresenham",
        )

        pred_kpts_global = pred_kpts.cpu().numpy().copy()
        pred_kpts_global[:, 0] += crop_l
        pred_kpts_global[:, 1] += crop_t

        output = {
            "average_brightness_head_to_tail": average_brightness_head_to_tail,
            "peak_heatmap_brightness_0": peak_heatmap_brightness_0,
            "peak_heatmap_brightness_1": peak_heatmap_brightness_1,
            "pred_kpts": pred_kpts.cpu().numpy(),
            "pred_kpts_global_px": pred_kpts_global,
        }
        outputs.append(output)
    return outputs


def get_pred_from_dir(
    crop_info,
    dataset,
    model,
    crop_after_model,
    pxl_to_cm_scale,
    vel_window_size,
    length_window_size,
    device,
    cone_eq_params_left=None,
    cone_eq_params_right=None,
    vel_delta_tolerance=15,
    length_delta_tolerance=5,
    min_edge_dist_tolerance=10,
    model_input_channels=1,
    mapTokpt_differentiable=False,
    mapTokpt_round_to_integer=False,
):

    model.eval()

    ml, bl = cone_eq_params_left
    mr, br = cone_eq_params_right

    all_fish_ids = set(
        [fish["fish_id"] for frame in crop_info for fish in frame["frame_crop_infos"]]
    )

    frame_nums = {fish_id: [] for fish_id in all_fish_ids}
    pred_lens_cm = {fish_id: [] for fish_id in all_fish_ids}
    pred_kpts_global_0_px = {fish_id: [] for fish_id in all_fish_ids}
    pred_kpts_global_1_px = {fish_id: [] for fish_id in all_fish_ids}
    pred_kpts_global_0_cm = {fish_id: [] for fish_id in all_fish_ids}
    pred_kpts_global_1_cm = {fish_id: [] for fish_id in all_fish_ids}
    min_edge_distances_pxl = {fish_id: [] for fish_id in all_fish_ids}
    av_brightnesses = {fish_id: [] for fish_id in all_fish_ids}
    peak_heatmap_brightnesses_0 = {fish_id: [] for fish_id in all_fish_ids}
    peak_heatmap_brightnesses_1 = {fish_id: [] for fish_id in all_fish_ids}
    pred_kpts = {fish_id: [] for fish_id in all_fish_ids}
    len_outputs = {}

    with torch.no_grad():
        for frame_crop_info in tqdm(crop_info):
            frame_num = frame_crop_info["frame_num"]

            print(f"{frame_num=} {frame_crop_info=}")
            if frame_crop_info["frame_crop_infos"] == []:
                continue

            fish_ids = [
                fish_crop_info["fish_id"]
                for fish_crop_info in frame_crop_info["frame_crop_infos"]
            ]
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
                frames_to_load = [
                    min(len(dataset) - 1, frame_num + 2),
                    frame_num,
                    min(len(dataset) - 1, frame_num + 1),
                ]

            elif frame_num == len(dataset) - 1:
                # MAH 2025-11-24 14:26:35 this isnt ideal as not how its trained, should probably just skip the first and last frames
                frames_to_load = [
                    max(0, frame_num - 1),
                    frame_num,
                    max(0, frame_num - 2),
                ]
            else:
                frames_to_load = [
                    max(0, frame_num - 1),
                    frame_num,
                    min(len(dataset) - 1, frame_num + 1),
                ]

            print(
                f"# MAH 2025-11-24 18:49:39 make this more efficient by using a get_item_ that can pull in multiple frames at once"
            )
            # MAH 2025-11-24 18:48:13 this should all be batched together to save time a get_item_ that can pull in multiple frames at once
            frame_images_previous, _frame_labels, _unwarped_frames, _echogram = (
                dataset.__getitem__(frames_to_load[0], postprocess=False)
            )
            frame_images_current, _frame_labels, _unwarped_frames, _echogram = (
                dataset.__getitem__(frames_to_load[1], postprocess=False)
            )
            frame_images_next, _frame_labels, _unwarped_frames, _echogram = (
                dataset.__getitem__(frames_to_load[2], postprocess=False)
            )

            frame_images_previous_bgs = torch.from_numpy(
                frame_images_previous[0][:, :, 1]
            ).float()
            frame_images_current_bgs = torch.from_numpy(
                frame_images_current[0][:, :, 1]
            ).float()
            frame_images_next_bgs = torch.from_numpy(
                frame_images_next[0][:, :, 1]
            ).float()

            frame_images_previous_bgs -= 255 / 2
            frame_images_current_bgs -= 255 / 2
            frame_images_next_bgs -= 255 / 2
            frame_images_previous_bgs[frame_images_previous_bgs < 0] = 0
            frame_images_current_bgs[frame_images_current_bgs < 0] = 0
            frame_images_next_bgs[frame_images_next_bgs < 0] = 0
            frame_images_previous_bgs /= torch.max(frame_images_previous_bgs)
            frame_images_current_bgs /= torch.max(frame_images_current_bgs)
            frame_images_next_bgs /= torch.max(frame_images_next_bgs)
            img = torch.stack(
                [
                    frame_images_previous_bgs,
                    frame_images_current_bgs,
                    frame_images_next_bgs,
                ],
                axis=0,
            )  # MAH 2025-11-24 12:23:12 I trained the detector on BGS (previous frame, current frame, next frame) so we need to pass in the previous, current, and next frames
            img = img.unsqueeze(
                0
            )  # MAH 2025-11-24 14:44:42 add batch dimension this might be able to be actually batched properly to do multiple at once?
            img = img.to(device)
            pred_infos = get_pred_from_img(
                img,
                model,
                crops_l=crop_ls,
                crops_t=crop_ts,
                crops_r=crop_rs,
                crops_b=crop_bs,
                model_input_channels=model_input_channels,
                mapTokpt_differentiable=mapTokpt_differentiable,
                mapTokpt_round_to_integer=mapTokpt_round_to_integer,
            )

            for fish_id, pred_info in zip(fish_ids, pred_infos):
                frame_nums[fish_id].append(frame_num)
                pred_lens_cm[fish_id].append(
                    calc_len(pred_info["pred_kpts"]) * pxl_to_cm_scale
                )
                pred_kpts_global_0_px[fish_id].append(
                    pred_info["pred_kpts_global_px"][0]
                )
                pred_kpts_global_1_px[fish_id].append(
                    pred_info["pred_kpts_global_px"][1]
                )
                pred_kpts_global_0_cm[fish_id].append(
                    pred_info["pred_kpts_global_px"][0] * pxl_to_cm_scale
                )
                pred_kpts_global_1_cm[fish_id].append(
                    pred_info["pred_kpts_global_px"][1] * pxl_to_cm_scale
                )
                min_edge_distances_pxl[fish_id].append(
                    get_min_edge_distances_pxl(
                        pred_info["pred_kpts_global_px"], ml, bl, mr, br
                    )[0]
                )
                av_brightnesses[fish_id].append(
                    pred_info["average_brightness_head_to_tail"]
                )
                peak_heatmap_brightnesses_0[fish_id].append(
                    pred_info["peak_heatmap_brightness_0"]
                )
                peak_heatmap_brightnesses_1[fish_id].append(
                    pred_info["peak_heatmap_brightness_1"]
                )

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
                abs(change) < length_delta_tolerance for change in change_in_length
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


# return {
#     "pred_lens": pred_lens,
#     "pred_kpts_global_0_px": pred_kpts_global_0_px,
#     "pred_kpts_global_1_px": pred_kpts_global_1_px,
#     "min_edge_distances_pxl": min_edge_distances_pxl,
#     "peak_heatmap_brightnesses_0": peak_heatmap_brightnesses_0,
#     "peak_heatmap_brightnesses_1": peak_heatmap_brightnesses_1,
#     "av_brightnesses": av_brightnesses,
#     "pred_kpts": pred_kpts,
#     "change_in_length": change_in_length,
#     "average_length": average_length,
#     "velocity_deviations": velocity_deviations,
# }

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
        crop_ltrbs=None,
    ):
        crop = crop_ltrbs is not None

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
        for crop_ltrb in crop_ltrbs:
            print(f"{crop_ltrb=}")

            pred_cropped = pred[
                :, :, crop_ltrb[1] : -crop_ltrb[3], crop_ltrb[0] : -crop_ltrb[2]
            ]
            img_cropped = img[
                :, :, crop_ltrb[1] : -crop_ltrb[3], crop_ltrb[0] : -crop_ltrb[2]
            ]

            pred_kpts = mapTokpt(
                pred_cropped,
                differentiable=self.mapTokpt_differentiable,
                round_to_integer=self.mapTokpt_round_to_integer,
            )[0]

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
            pred_kpts_global[:, 0] += crop_ltrb[0]
            pred_kpts_global[:, 1] += crop_ltrb[1]

            output = {
                # "pred_kpts": pred_kpts.cpu().numpy(),
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

        len_outputs = {}

        with torch.no_grad():
            for frame_crop_info in tqdm(crop_info):
                frame_num = frame_crop_info["frame_num"]

                if frame_crop_info["crop_ltrbs"] == []:
                    continue
                crop_ltrbs = frame_crop_info["crop_ltrbs"]
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

                print(f"taking the middle (bgs) channel")
                frames_bgs = frames[:, 1]

                frames_bgs -= 255 / 2
                frames_bgs[frames_bgs < 0] = 0
                frames_bgs /= torch.max(frames_bgs)
                frames_bgs = frames_bgs.unsqueeze(0)
                # frames_bgs = frames_bgs.to(device)
                pred_infos = self.get_pred_from_img(
                    frames_bgs,
                    crop_ltrbs=crop_ltrbs,
                )
                len_outputs[frame_num] = pred_infos

        return len_outputs

    def run(self, frames_batch, pred_bboxes):
        return self.get_length_estimates(frames_batch, pred_bboxes)

    def get_crop_info(self, pred_bboxes_batch):
        crop_info = []
        for batch_idx_frame_idx, frame_pred_bboxes in pred_bboxes_batch.items():
            crop_ltrbs = []
            frame_idx = batch_idx_frame_idx[1]

            if frame_pred_bboxes is None:
                continue

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

                crop_ltrbs.append([crop_l, crop_t, crop_r, crop_b])
            crop_info.append(
                {
                    "frame_num": frame_idx,
                    "crop_ltrbs": crop_ltrbs,
                }
            )
        return crop_info

    def get_length_estimates(self, frames_batch, pred_bboxes_batch):
        crop_info = self.get_crop_info(pred_bboxes_batch)

        pred_len_outputs = self.get_pred_from_batch(
            crop_info,
            frames_batch,
        )
        return pred_len_outputs

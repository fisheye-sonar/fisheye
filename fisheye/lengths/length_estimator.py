import torch
from math import floor, ceil
from fisheye.lengths.length_models import get_model
from fisheye.lengths.measure_utils import get_cone_edges

from fisheye.lengths.measure_utils import (
    get_velocity_dev,
    get_change_in_length,
    mapTokpt,
    average_brightness_on_line,
)
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from fisheye.lengths.measure_utils import vis_3_channel_img


class LengthEstimator:
    def __init__(self, metadata):
        self.metadata = metadata

        self.xdim = metadata.xdim
        self.ydim = metadata.ydim

        model_type = "unet"
        unet_double_conv = False
        self.model_input_channels = 1
        self.model_input_channels = 3
        load_model_path = "/home/mahobley/Code/fisheye-dev/head_tail/checkpoints/crop_after_model/model_150.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.crop_after_model = (
            True  # set true is expect to have multiple crops for each frame
        )
        self.padd_for_receptive_field = 100  # this is the amount of padding to add to the crop to remove edge effects of the receptive field when cropping before the model

        self.additional_bbox_padding_px = 25  # this is the padding to compensate for the bbox being sometimes being slightly smaller than the actual fish

        self.vel_window_size = 7
        self.length_window_size = 7
        self.vel_delta_tolerance = None
        self.length_delta_tolerance = None
        self.min_edge_dist_tolerance = 10
        self.min_edge_dist_tolerance = None

        self.mapTokpt_differentiable = False
        self.mapTokpt_round_to_integer = False

        self.return_average_brightness_head_to_tail = False
        self.return_peak_heatmap_brightnesses = False

        self.plot_pred_kpts = False

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

    def get_pred_data_from_cropped_pred(self, pred_cropped, img_cropped, crop_ltrb):
        pred_kpts = mapTokpt(
            pred_cropped,
            differentiable=self.mapTokpt_differentiable,
            round_to_integer=self.mapTokpt_round_to_integer,
        )[0]

        # print(f"{i=} {pred_kpts.cpu().numpy().tolist()} {pred_cropped.shape}")

        if self.plot_pred_kpts:
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(pred_cropped[0, 0].cpu().numpy())
            ax[1].imshow(pred_cropped[0, 1].cpu().numpy())
            ax[2].imshow(vis_3_channel_img(img_cropped[0]).cpu().numpy())
            ax[2].scatter(
                pred_kpts[:, 0].cpu().numpy(),
                pred_kpts[:, 1].cpu().numpy(),
                color="green",
                marker="x",
            )
            plt.savefig(
                f"pred_kpts_{crop_ltrb[0]}_{crop_ltrb[1]}_{crop_ltrb[2]}_{crop_ltrb[3]}.png"
            )
            plt.show()

        pred_kpts_global = pred_kpts.cpu().numpy().copy()
        pred_kpts_global[:, 0] += crop_ltrb[0]
        pred_kpts_global[:, 1] += crop_ltrb[1]

        output = {
            # "pred_kpts": pred_kpts.cpu().numpy(),
            "pred_kpts_global_px": pred_kpts_global,
        }
        if self.return_average_brightness_head_to_tail:
            average_brightness_head_to_tail = average_brightness_on_line(
                img_cropped[0, 1].cpu().numpy(),
                pred_kpts[0].cpu().numpy(),
                pred_kpts[1].cpu().numpy(),
                method="bresenham",
            )
            output["average_brightness_head_to_tail"] = average_brightness_head_to_tail
        if self.return_peak_heatmap_brightnesses:
            peak_heatmap_brightness_0 = torch.max(pred_cropped[:, 0]).item()
            peak_heatmap_brightness_1 = torch.max(pred_cropped[:, 1]).item()
            output["peak_heatmap_brightness_0"] = peak_heatmap_brightness_0
            output["peak_heatmap_brightness_1"] = peak_heatmap_brightness_1

        return output

    def get_pred_from_img(self, img, crop_ltrbs=None):

        if self.model_input_channels == 1:
            if img.shape[1] == 3:
                # if given a 3 channel image (prev, current, next) we want to use the current channel
                img = img[:, 1:2, :, :]
        self.model.eval()

        outputs = []
        if self.crop_after_model:
            # whole image
            pred = self.model(img.float())

            for crop_ltrb in crop_ltrbs:

                pred_cropped = pred[
                    :, :, crop_ltrb[1] : -crop_ltrb[3], crop_ltrb[0] : -crop_ltrb[2]
                ]
                img_cropped = img[
                    :, :, crop_ltrb[1] : -crop_ltrb[3], crop_ltrb[0] : -crop_ltrb[2]
                ]

                outputs.append(
                    self.get_pred_data_from_cropped_pred(
                        pred_cropped,
                        img_cropped,
                        crop_ltrb,
                    )
                )
        else:
            for crop_ltrb in crop_ltrbs:
                min_crop_l = max(0, crop_ltrb[0] - self.padd_for_receptive_field)
                min_crop_t = max(0, crop_ltrb[1] - self.padd_for_receptive_field)
                min_crop_r = max(0, crop_ltrb[2] - self.padd_for_receptive_field)
                min_crop_b = max(0, crop_ltrb[3] - self.padd_for_receptive_field)

                amount_padded_left = crop_ltrb[0] - min_crop_l
                amount_padded_top = crop_ltrb[1] - min_crop_t
                amount_padded_right = crop_ltrb[2] - min_crop_r
                amount_padded_bottom = crop_ltrb[3] - min_crop_b

                img_init_crop = img[
                    :,
                    :,
                    min_crop_t : img.shape[2] - min_crop_b,
                    min_crop_l : img.shape[3] - min_crop_r,
                ]

                pred_init_crop = self.model(img_init_crop)
                # remove receptive field padding
                pred_cropped = pred_init_crop[
                    :,
                    :,
                    amount_padded_top:-amount_padded_bottom,
                    amount_padded_left:-amount_padded_right,
                ]
                img_cropped = img_init_crop[
                    :,
                    :,
                    amount_padded_top:-amount_padded_bottom,
                    amount_padded_left:-amount_padded_right,
                ]
                img_cropped2 = img[
                    :,
                    :,
                    crop_ltrb[1] : -crop_ltrb[3],
                    crop_ltrb[0] : -crop_ltrb[2],
                ]
                # MAH 2025-11-26 11:39:45 checking its the same as as if i did the crop after the model
                if not torch.allclose(img_cropped, img_cropped2, atol=1e-6):
                    print("crops different")
                outputs.append(
                    self.get_pred_data_from_cropped_pred(
                        pred_cropped,
                        img_cropped,
                        crop_ltrb,
                        i=i,
                    )
                )

        return outputs

    def get_pred_from_batch(
        self,
        crop_info,
        frames_batch,
        # device,
    ):

        len_outputs = {}

        with torch.no_grad():
            # for frame_crop_info in tqdm(crop_info):
            for frame_crop_info in crop_info:
                frame_num = frame_crop_info["frame_num"]

                if frame_crop_info["crop_ltrbs"] == []:
                    continue
                crop_ltrbs = frame_crop_info["crop_ltrbs"]
                # MAH 2025-11-26 10:55:53 this works on the batch so the forst and last fram eof every batch is going to be wrong, need to have some kind of batch overlap or the dataloader gets the next and last frame anyway
                # MAH 2025-11-24 14:26:35 this isnt ideal as not how its trained, some of these are getting 2 of the same frames on 2 channels

                prev_ind = max(0, frame_num - 1)
                next_ind = min(len(frames_batch) - 1, frame_num + 1)
                frames = torch.stack(
                    [
                        frames_batch[prev_ind],
                        frames_batch[frame_num],
                        frames_batch[next_ind],
                    ],
                    dim=0,
                )

                # MAH 2025-11-26 10:57:42 this is taking the middle (bgs) channel maybe we try a different training strategy that aligns with our existing pipeline
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

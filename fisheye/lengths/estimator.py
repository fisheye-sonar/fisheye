from math import floor, ceil

import torch
from matplotlib import pyplot as plt

from fisheye.configs.models import BaseLengthModelConfig, UNetLengthModelConfig
from fisheye.lengths.base import BaseLengthEstimator
from fisheye.lengths.models import get_model
from fisheye.lengths.measure_utils import get_cone_edges
from fisheye.lengths.measure_utils import (
    mapTokpt,
    average_brightness_on_line,
)
from fisheye.lengths.measure_utils import vis_3_channel_img


class UNetLengthEstimator(BaseLengthEstimator):
    def __init__(self, metadata, config: BaseLengthModelConfig = None):
        super().__init__(metadata)

        if config is None:
            config = UNetLengthModelConfig()
        self.config = config

        self.xdim = metadata.xdim
        self.ydim = metadata.ydim

        self.model_input_channels = self.config.input_channels

        self.crop_after_model = self.config.crop_after_model
        self.padd_for_receptive_field = self.config.padd_for_receptive_field

        self.additional_bbox_padding_px = self.config.additional_bbox_padding_px

        self.mapTokpt_differentiable = False
        self.mapTokpt_round_to_integer = False

        self.return_average_brightness_head_to_tail = False
        self.return_peak_heatmap_brightnesses = False

        self.plot_pred_kpts = False

        self.bgs_3_channel = self.config.bgs_3_channel

        self.model = get_model(
            model_type=self.config.type,
            model_input_channels=self.config.input_channels,
            unet_double_conv=self.config.unet_double_conv,
            weights=self.config.weights,
            device=self.config.device,
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
                # add padding so the model doesnt see the edge effects of the receptive field
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

                outputs.append(
                    self.get_pred_data_from_cropped_pred(
                        pred_cropped,
                        img_cropped,
                        crop_ltrb,
                    )
                )

        return outputs

    def get_pred_from_batch(self, crop_info, frames_batch, bgs_3_channel=True):

        len_outputs = {}

        with torch.no_grad():
            for frame_crop_info in crop_info:
                frame_num = frame_crop_info["frame_num"]

                if frame_crop_info["crop_ltrbs"] == []:
                    continue
                crop_ltrbs = frame_crop_info["crop_ltrbs"]
                # MAH 2025-11-26 10:55:53 this works on the batch so the first and last frame of every batch is going
                # to be wrong, need to have some kind of batch overlap or the dataloader gets the next and last frame
                # anyway

                prev_ind = max(0, frame_num - 1)
                next_ind = min(len(frames_batch) - 1, frame_num + 1)

                if bgs_3_channel:
                    frames = torch.stack(
                        [
                            frames_batch[prev_ind],
                            frames_batch[frame_num],
                            frames_batch[next_ind],
                        ],
                        dim=0,
                    )

                    # MAH 2025-11-26 10:57:42 this is taking the middle (bgs) channel maybe we try a different training
                    # strategy that aligns with our existing pipeline
                    frames_bgs = frames[:, 1]

                    frames_bgs -= 255 / 2
                    frames_bgs[frames_bgs < 0] = 0
                    frames_bgs /= torch.max(frames_bgs)
                    frames_bgs = frames_bgs.unsqueeze(0)
                    frames_to_run = frames_bgs
                else:
                    # MAH 2025-12-19 16:11:40 could be that we can actually run the model on all the frames in the batch in one go not in a for loop
                    frames_to_run = frames_batch[frame_num].unsqueeze(0)

                pred_infos = self.get_pred_from_img(
                    frames_to_run,
                    crop_ltrbs=crop_ltrbs,
                )
                len_outputs[frame_num] = pred_infos

        return len_outputs

    def run(self, frames_batch, pred_bboxes):
        """Run the model on a batch of frames and predicted bounding boxes."""
        return self.get_length_estimates(
            frames_batch, pred_bboxes, bgs_3_channel=self.config.bgs_3_channel
        )

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

    def get_length_estimates(self, frames_batch, pred_bboxes_batch, bgs_3_channel=True):
        """Get length estimates from the model for a batch of frames and predicted bounding boxes."""
        crop_info = self.get_crop_info(pred_bboxes_batch)

        pred_len_outputs = self.get_pred_from_batch(
            crop_info,
            frames_batch,
            bgs_3_channel=bgs_3_channel,
        )
        return pred_len_outputs

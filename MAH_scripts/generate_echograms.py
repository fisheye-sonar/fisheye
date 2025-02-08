import pytest
import sys
import os

sys.path.append("/Users/mahobley/Code/fisheye")
from fisheye.dataloaders.aris import create_aris_dataloader
import matplotlib.pyplot as plt
import numpy as np
from MAH_utils import (
    make_gif_from_np_stack,
    numpy_to_redblue,
    zero_pad_to_match_one_dim,
)

import numpy as np
import pytest
import torch

from fisheye.dataloaders import     create_aris_dataloader

from fisheye.config import ARISDatasetConfig


ARIS_FILE = "/Users/mahobley/Downloads/2024-10-28_113000.aris"


def make_echogram_image(echograms, echogram_pop=False):

    ec_mag = echograms[:, :, 0]  # magnitude of the echogram
    ec_angle = echograms[:, :, 1]  # angle of the echogram

    ec_mag -= ec_mag.min()
    ec_mag /= ec_mag.max()

    # find the areas of the echogram that are brighter than the average for that row
    ec_mag_bgs = ec_mag - np.mean(ec_mag, axis=0)
    ec_mag_bgs[ec_mag_bgs < 0] = 0
    ec_mag_bgs /= ec_mag_bgs.max()

    colmapped = numpy_to_redblue(((ec_angle) * ec_mag_bgs) * 2)

    if echogram_pop:
        # take the average subtracted echogram
        stacked_echogram_image = np.stack([ec_mag_bgs, ec_mag_bgs, ec_mag_bgs], axis=-1)
    else:
        # take the raw echogram
        stacked_echogram_image = np.stack([ec_mag, ec_mag, ec_mag], axis=-1)

    colour_mask = np.stack([ec_mag_bgs, ec_mag_bgs, ec_mag_bgs], axis=-1)
    colour_mask[colour_mask < 0.25] = 0
    colour_mask[colour_mask > 0] = 1

    output_image = stacked_echogram_image * (1 - colour_mask) + colmapped * (
        colour_mask
    )
    output_image = output_image.transpose(1, 0, 2)

    # make it a video
    output_image = np.stack([output_image] * output_image.shape[1])
    output_image = (output_image * 253).astype(np.uint8)

    # add the vertical white line
    for i in range(output_image.shape[0]):
        output_image[i, :, i] = 254

    return output_image



def generate_echogram_gif_from_aris(config, save_filename, echogram_pop):
    dataloader, dataset = create_aris_dataloader(config)

    echograms = []
    frames_vis = []
    for i, batch in enumerate(dataset):
        frames, unwarped_frames, echogram = (
            batch[0],
            batch[2],
            batch[3],
        )
        print(i)
        print(type(frames))
        print(type(unwarped_frames))
        print(type(echogram))
        frames_vis.append(frames)
        echograms.append(echogram)
    echograms = np.concatenate(echograms, axis=0)
    frames_vis = np.concatenate(frames_vis, axis=0)

    coloured_echogram = make_echogram_image(echograms, echogram_pop=echogram_pop)

    coloured_echogram = coloured_echogram[: frames_vis.shape[0]]
    coloured_echogram = zero_pad_to_match_one_dim(coloured_echogram, frames_vis.shape, dim=1)
    frames_vis = np.stack([frames_vis[:, :, :, 0]] * 3, axis=-1)
    if coloured_echogram.shape[2] < frames_vis.shape[2]:
        coloured_echogram = np.repeat(
            coloured_echogram, int(frames_vis.shape[2] / coloured_echogram.shape[2]), axis=2
        )
    comb = np.concatenate([frames_vis, coloured_echogram], axis=2)
    print("Saving gif...")
    make_gif_from_np_stack(save_filename, comb, frame_rate=25)


fp = "/Users/mahobley/Code/salmon_counting_data/RO_2018-05-26_073004.aris"
fp = "/Users/mahobley/Downloads/2024-10-28_113000.aris"
beam_width_dir = "/Users/mahobley/Code/salmon_counting_data/beam_widths"

config = ARISDatasetConfig(aris_filepath=fp,
    beam_width_dir=beam_width_dir,
    return_unwarped=True,
    return_echogram=True,
    start_frame=181,
    end_frame=188,
)

generate_echogram_gif_from_aris(
    config,
    save_filename="_debugging_images/test13.gif",
    echogram_pop=True,
)

import pytest
import sys
import os

sys.path.append("/Users/mahobley/Code/fisheye")
from fisheye.dataloaders.aris import create_aris_dataloader
import matplotlib.pyplot as plt
import numpy as np
from MAH_utils import (
    make_gif_from_np_stack,
)

import numpy as np
import pytest
import torch

from fisheye.dataloaders import create_aris_dataloader
from generate_echograms import (
    make_echogram_image,
    zero_pad_to_match_one_dim,
)
from fisheye.config import ARISDatasetConfig


def generate_echogram_gif_from_aris(
    config, save_filename, echogram_pop, return_unwarped
):
    dataloader, dataset = create_aris_dataloader(config)

    echograms = []
    frames_vis = []
    for i, batch in enumerate(dataset):
        frames, echogram = (
            batch[0],
            batch[3],
        )
        print(f"{i=} {frames.shape=} {echogram.shape=}")
        if frames.shape[0] != 0:
            frames_vis.append(frames[:1])
            echograms.append(echogram[:1])
    echograms = np.concatenate(echograms, axis=0)
    frames_vis = np.concatenate(frames_vis, axis=0)
    print(f"{frames_vis.shape=} {echograms.shape=} ")

    coloured_echogram = make_echogram_image(echograms.astype('float'), echogram_pop=echogram_pop)

    coloured_echogram = coloured_echogram[: frames_vis.shape[0]]
    coloured_echogram = zero_pad_to_match_one_dim(
        coloured_echogram, frames_vis.shape, dim=1
    )
    frames_vis = np.stack([frames_vis[:, :, :, 0]] * 3, axis=-1)
    if coloured_echogram.shape[2] < frames_vis.shape[2]:
        coloured_echogram = np.repeat(
            coloured_echogram,
            int(frames_vis.shape[2] / coloured_echogram.shape[2]),
            axis=2,
        )
    comb = np.concatenate([frames_vis, coloured_echogram], axis=2)
    print("Saving gif...")
    make_gif_from_np_stack(save_filename, comb, frame_rate=25)


fp = "/Users/mahobley/Code/salmon_counting_data/RO_2018-05-26_073004.aris"
# fp = "/Users/mahobley/Downloads/2024-10-28_113000.aris"
beam_width_dir = "/Users/mahobley/Code/salmon_counting_data/beam_widths"

config = ARISDatasetConfig(
    aris_filepath=fp,
    beam_width_dir=beam_width_dir,
    return_unwarped=False,
    return_echogram=True,
    start_frame=0,
    end_frame=150,
)

generate_echogram_gif_from_aris(
    config,
    save_filename="_debugging_images/test15.gif",
    echogram_pop=True,
    return_unwarped=False,
)

import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import sys

sys.path.append("/home/mahobley/Code/fisheye")
sys.path.append("/home/mahobley/Code/fisheye-dev")

from fisheye.dataloaders.didson.pyDIDSON import DIDSON

from utilities.col_console_text import col


def generate_echogram_from_aris(
    aris_path,
    start_frame=None,
    end_frame=None,
    bgs=True,
    return_echogram_with_how_wide_the_peak_as_third_channel=True,
    return_echogram_with_no_bgs_as_third_channel=False,
    return_echogram_with_distances_as_third_channel=False,
    debug_plot=False,
):
    """
    Processes DIDSON/ARIS files by extracting raw frames and saving them as JPEG images.

    Function will run through directories looking for DIDSON/ARIS files. Once frames have been extracted,
    a new directory (name of DIDSON/ARIS file) will be created to store the raw frames.
    Args:
        root_dir (str): Path to the root directory containing subdirectories with DIDSON/ARIS files.
        output_dir (str): Path to the directory where frames will be saved.
        beam_width_dir (str): Path to the directory with beam width information.
    """

    if aris_path.endswith(".ddf") or aris_path.endswith(".aris"):
        didson_file_path = aris_path
        print(f"Processing file: {didson_file_path}")

        didson = DIDSON(didson_file_path)
        frame_start = 0 if start_frame is None else start_frame
        frame_end = -1 if end_frame is None else end_frame

        if (
            return_echogram_with_how_wide_the_peak_as_third_channel
            or return_echogram_with_distances_as_third_channel
        ):
            print(f"{col.green}Loading beam width information...{col.end}")
            info = didson.info
            samplesperbeam = info["samplesperbeam"]
            windowlength = info["windowlength"]
            windowstart = info["windowstart"]

            bin_distance = np.empty(samplesperbeam)
            for i in range(samplesperbeam):
                bin_distance[samplesperbeam - i - 1] = (
                    windowstart + i * windowlength / samplesperbeam
                )
            print(f"{col.green}Loaded beam width information...{col.end}")

        echogram = didson.load_echogram(
            start_frame=frame_start,
            end_frame=frame_end,
            num_frames_bg_subtract=100000,
            use_blur=False,
            return_echogram_with_bg_subtracted=bgs,
            return_echogram_with_how_wide_the_peak_as_third_channel=return_echogram_with_how_wide_the_peak_as_third_channel,
            return_echogram_with_no_bgs_as_third_channel=return_echogram_with_no_bgs_as_third_channel,
            return_echogram_with_distances_as_third_channel=return_echogram_with_distances_as_third_channel,
        )

        echogram = echogram.transpose(1, 0, 2)
        echogram = np.nan_to_num(echogram, nan=0.0, posinf=0.0, neginf=0.0)
        # print(f"{echogram.shape=} {np.min(echogram)=} {np.max(echogram)=}")
        # add a black boundary to the echograms
        echogram[:, 0, :] = 0
        echogram[:, -1, :] = 0
        echogram[0, :, :] = 0
        echogram[-1, :, :] = 0

        height, num_frames, _ = echogram.shape
        if echogram.shape[2] > 3:
            print(
                f"{col.red}Warning: echogram has more than 3 channels, only the first 3 will be saved{col.end}"
            )
        if return_echogram_with_how_wide_the_peak_as_third_channel:
            echogram[:, :, 2] = echogram[:, :, 2] * bin_distance[:, None]
            echogram[:, :, 2] = echogram[:, :, 2] / 255.0
        if return_echogram_with_distances_as_third_channel:
            echogram[:, :, 2] = (
                bin_distance[:, None] - bin_distance[-1]
            ) / 35.0  # normalise by max distance of 35m

        if debug_plot:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            for c in range(echogram.shape[2]):
                ax[c].imshow(echogram[:, :, c])
                ax[c].set_title(f"Channel {c}")
            plt.show()

        # MAH 2026-02-06 13:22:02 TODO save with start and end frame in name
        rgb = np.zeros((height, num_frames, 3), dtype=np.float32)
        rgb[:, :, 0] = echogram[:, :, 0] * 255  # Red channel: max values
        rgb[:, :, 1] = (
            echogram[:, :, 1] + 0.5
        ) * 255  # Green channel: argmax positions

        if echogram.shape[2] > 2:
            rgb[:, :, 2] = echogram[:, :, 2] * 255  # Blue channel: max values
        rgb = np.clip(rgb, 0, 255)
        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # ax[0].imshow(rgb[:, :, 0])  # Transpose for correct orientation
        # ax[1].imshow(rgb[:, :, 1])  # Transpose for correct orientation
        # ax[2].imshow(rgb[:, :, 2])  # Transpose for correct orientation
        # plt.show()
        rgb = rgb.astype(np.uint8)
        # return the rgb image
        return rgb


if __name__ == "__main__":

    start_frame = 3642
    end_frame = 3842
    aris_path = "/home/mahobley/Data/CFC22/aris_files/kenai-rightbank/2018-05-26-JD146_RightFar_Stratum1_Set1_RO_2018-05-26_050004.aris"  # Path to the root directory containing subdirs with DIDSON files
    save_name = "2018-05-26-JD146_RightFar_Stratum1_Set1_RO_2018-05-26_050004_3642_3842"
    start_frame = 3347
    end_frame = 3547
    aris_path = "/home/mahobley/Data/CFC22/aris_files/kenai-rightbank/2018-05-27-JD147_RightFar_Stratum1_Set1_RO_2018-05-27_220004.aris"  # Path to the root directory containing subdirs with DIDSON files
    save_name = "2018-05-27-JD147_RightFar_Stratum1_Set1_RO_2018-05-27_220004_3347_3547"
    start_frame = 0
    end_frame = -1
    aris_path = "/mnt/data/CFC26_MAH/aris/kenai-rightbank-stratum1/2018-05-27-JD147_RightFar_Stratum1_Set1_RO_2018-05-27_220004.aris"  # Path to the root directory containing subdirs with DIDSON files
    aris_path = "/mnt/data/CFC26_MAH/aris/kenai-rightbank-stratum1/2018-06-10-JD161_RightNear_Stratum1_Set1_RN_2018-06-10_030003.aris"  # Path to the root directory containing subdirs with DIDSON files
    aris_path = "/mnt/data/CFC26_MAH/aris/klamath/2024-11-08_140000.aris"  # Path to the root directory containing subdirs with DIDSON files
    save_name = f"{aris_path.split('/')[-1].split('.')[0]}_{start_frame}_{end_frame}"

    output_dir_ims = "/home/mahobley/Code/fisheye-dev/echograms/echograms_ARIS_ims"  # Path to save extracted frames
    output_dir_ims = "/home/mahobley/Code/echo-seg/"  # Path to save extracted frames
    output_dir_np = "/home/mahobley/Code/fisheye-dev/echograms/echograms_ARIS_np"  # Path to save extracted frames
    output_dir_np = None  # Path to save extracted frames
    beam_width_dir = "/home/mahobley/Code/fisheye_main/fisheye/beam_widths"  # Path to beam width information
    bgs = True

    # 2018-06-10-JD161_RightNear_Stratum1_Set1_RN_2018-06-10_190004_00968_01168_predictions_cropped

    return_echogram_with_how_wide_the_peak_as_third_channel = False
    return_echogram_with_no_bgs_as_third_channel = True
    return_echogram_with_distances_as_third_channel = False

    # time this
    import time

    start_time = time.time()
    # 2018-05-27-JD147_RightFar_Stratum1_Set1_RO_2018-05-27_220004_3347_3547
    echoogram = generate_echogram_from_aris(
        aris_path,
        start_frame=start_frame,
        end_frame=end_frame,
        bgs=bgs,
        return_echogram_with_how_wide_the_peak_as_third_channel=return_echogram_with_how_wide_the_peak_as_third_channel,
        return_echogram_with_no_bgs_as_third_channel=return_echogram_with_no_bgs_as_third_channel,
        return_echogram_with_distances_as_third_channel=return_echogram_with_distances_as_third_channel,
        debug_plot=False,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"{echoogram.shape=}")

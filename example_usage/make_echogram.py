import os

from fisheye.dataclasses import ARISDatasetConfig
from utils.visualisation_utils import (
    generate_echogram_vis_from_aris,
    make_gif_from_np_stack,
    make_vid_from_np_stack,
)
from PIL import Image

config = ARISDatasetConfig(
    filepath="/Users/mahobley/Code/salmon_counting_data/2024-10-28_113000.aris",
    start_frame=181,
    end_frame=313,
    batch_size=1,
    return_unwarped=False,
    return_echogram=True,
    echogram_filter_kernel=7,
    echogram_filter_tol=0.15,
)

list_frames = generate_echogram_vis_from_aris(
    config,
    echogram_pop=True,
    return_unwarped=False,
    resize_mode="scale",  # scale or pad
    return_list=True,
    colour_image_edges=True,
)


filename = "_debugging_images/"
filename += os.path.basename(config.filepath).split(".")[0]
filename += f"_{config.start_frame}-{config.end_frame}_{config.echogram_filter_kernel}_{int(config.echogram_filter_tol*100)}"

print("Saving video...")
make_vid_from_np_stack(filename + ".mp4", list_frames, frame_rate=12)

print("Saving image...")

im = Image.fromarray(list_frames[0])
im.save(filename + "_firstframe.jpeg")


# print("Saving gif...")
# make_gif_from_np_stack(
#     filename + ".gif",
#     list_frames,
#     frame_rate=25,
# )

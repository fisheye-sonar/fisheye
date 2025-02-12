import numpy as np
from PIL import Image

from fisheye.dataloaders.aris import create_aris_dataloader
from utils.generate_echograms import make_echogram_image, zero_pad_to_match_one_dim


def make_gif_from_np_stack(
    fn, fish_images_out_of_ordinary_vals, frame_rate=25, norm=False
):
    if norm:
        fish_images_out_of_ordinary_vals -= np.min(fish_images_out_of_ordinary_vals)
        fish_images_out_of_ordinary_vals /= np.max(fish_images_out_of_ordinary_vals)

    if np.max(fish_images_out_of_ordinary_vals) <= 1:
        scale_factor = 255
    else:
        scale_factor = 1
    pil_images = [
        Image.fromarray(np.uint8(img * scale_factor))
        for img in fish_images_out_of_ordinary_vals
    ]
    pil_images[0].save(
        fn, save_all=True, append_images=pil_images[1:], duration=1 / frame_rate, loop=0
    )
    print(f"GIF saved as {fn}")


def generate_echogram_gif_from_aris(
    config, save_filename, echogram_pop, return_unwarped
):
    dataloader, dataset = create_aris_dataloader(config)

    echograms = []
    frames_vis = []
    for i, batch in enumerate(dataloader):
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

    coloured_echogram = make_echogram_image(
        echograms.astype("float"), echogram_pop=echogram_pop
    )

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
    if save_filename:
        print("Saving gif...")
        make_gif_from_np_stack(save_filename, comb, frame_rate=25)

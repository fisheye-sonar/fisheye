import numpy as np
from PIL import Image


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

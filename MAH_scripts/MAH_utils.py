import numpy as np
from PIL import Image

def zero_pad_to_match_one_dim(array, target_shape, dim):
    """
    Zero-pads the input array to match the target shape in one dimension.

    Args:
        array (np.ndarray): The input array to be padded.
        target_shape (tuple): The target shape to pad the array to.
        dim (int): The dimension to pad.

    Returns:
        np.ndarray: The zero-padded array with the target shape in the specified dimension.
    """
    pad_width = [(0, 0)] * array.ndim
    pad_width[dim] = (0, max(0, target_shape[dim] - array.shape[dim]))
    padded_array = np.pad(array, pad_width, mode="constant", constant_values=0)
    return padded_array


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


def numpy_to_redblue(array):
    """
    Maps a value from -1 to 1 to an RGB color gradient transitioning
    from red (-1) to white (0) to blue (1).

    Args:
        value (float): A number between -1 and 1.

    Returns:
        tuple: (R, G, B) values as integers in the range [0, 255].
    """
    # Ensure the value is clamped between -1 and 1
    # value = max(-1, min(1, value))
    cmapped_pos = np.stack(
        [255 * np.ones_like(array), 255 * (1 + array), 255 * (1 + array)], axis=-1
    )
    cmapped_neg = np.stack(
        [
            255 * (1 - array),
            255 * (1 - array) * 0.5 + 128 * np.ones_like(array),
            255 * np.ones_like(array),
        ],
        axis=-1,
    )
    array_stacked = np.stack([array, array, array], axis=-1)
    cmapped = np.where(
        array_stacked <= 0,
        cmapped_pos,
        cmapped_neg,
    )
    cmapped = cmapped.astype(np.float32) / 255

    cmapped = np.clip(
        cmapped,
        0,
        1,
    )
    return cmapped

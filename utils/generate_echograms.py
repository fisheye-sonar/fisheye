import numpy as np
from scipy.signal import convolve


def make_echogram_image(echograms, echogram_pop=False, filter_kernel=0, filter_tol=0.5):
    """

    Args:
        echograms (numpy.ndarray): The exchogram data in [time, height, 2], the last dimension is the magnitude
                                of the echogram and the angle the max was found at
        echogram_pop (bool): whether to do background subtract (per row) on the echogram

    Returns:
        np.array: [time, height, 3] (R, G, B) values as integers in the range [0, 255] of the echogram,
                    with a colour scheme going from red to blue (left to right).
    """
    ec_mag = echograms[:, :, 0]  # magnitude of the echogram
    ec_angle = echograms[:, :, 1]  # angle of the echogram

    ec_mag -= ec_mag.min()
    ec_mag /= ec_mag.max()

    # find the areas of the echogram that are brighter than the average for that row
    ec_mag_bgs = ec_mag - np.mean(ec_mag, axis=0)
    ec_mag_bgs[ec_mag_bgs < 0] = 0
    ec_mag_bgs /= ec_mag_bgs.max()

    colmapped = numpy_to_redblue(
        ec_angle * 2
    )  # the squared makes the colours more extreme
    if filter_kernel != 0:
        kernel = np.ones((filter_kernel, filter_kernel))
        ec_mag_bgs_bin = np.where(ec_mag_bgs > filter_tol, 1, 0)
        neighbourhood = convolve(ec_mag_bgs_bin, kernel, mode="same")
        neighbourhood_bin = np.where(neighbourhood > 2, 1, 0.5)
        ec_mag_bgs *= neighbourhood_bin
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


def zero_pad_to_match_one_dim(array, target_shape, dim, centered=True, const_value=0):
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
    pad_ammount = max(0, target_shape[dim] - array.shape[dim])
    if centered:
        padd = (int(pad_ammount / 2), pad_ammount - int(pad_ammount / 2))
    else:
        padd = (0, pad_ammount)
    pad_width[dim] = padd
    padded_array = np.pad(
        array, pad_width, mode="constant", constant_values=const_value
    )
    return padded_array

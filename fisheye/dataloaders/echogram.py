import numpy as np

def compute_echogram(
    unwarped_frames,
    mean_blurred_frame=None,
    mean_normalization_value=None,
    return_echogram_with_bg_subtracted=True,
    return_raw_echogram_as_third_channel=False,
):
    """
    Generate an echogram from unwarped beam frames.
    Output channels:
    0: magnitude (max over bins)
    1: normalized argmax bin index in [-0.5, 0.5)
    2: (optional) magnitude without bg subtraction
    """

    num_channels = 3 if return_raw_echogram_as_third_channel else 2

    output = np.zeros(
        (*unwarped_frames.shape[:2], num_channels),
        dtype=np.float32,
    )

    frames_f32 = unwarped_frames.astype(np.float32)

    raw_echogram = None
    if return_raw_echogram_as_third_channel:
        raw_echogram = np.max(frames_f32, axis=2) / 255.0

    proc = frames_f32
    if return_echogram_with_bg_subtracted:
        if mean_blurred_frame is None or mean_normalization_value is None:
            raise ValueError(
                "mean_blurred_frame and mean_normalization_value are required when "
                "return_echogram_with_bg_subtracted=True"
            )
        proc = proc - mean_blurred_frame
        proc = proc / mean_normalization_value

    output[:, :, 0] = np.max(proc, axis=2)
    angle_echogram = np.argmax(proc, axis=2)
    depth = unwarped_frames.shape[2]
    col = angle_echogram.astype(np.float32) / float(depth)
    col -= 0.5
    output[:, :, 1] = col.astype(np.float32)

    if return_raw_echogram_as_third_channel:
        output[:, :, 2] = raw_echogram.astype(np.float32)


    return output

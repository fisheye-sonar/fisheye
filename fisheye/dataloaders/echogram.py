import cv2
import numpy as np

from fisheye.common.generic import run_with_threads


def compute_bg_subtraction(
    frames_for_bg_subtract,
    use_blur=True,
    use_multithreading=True,
    max_workers=2,
):
    """Calculate the mean blurred frame and normalization value for echogram bg subtraction."""
    if not use_blur:
        mean_blurred_frame = np.mean(frames_for_bg_subtract, axis=0)
        max_blurred_frame = np.max(np.abs(frames_for_bg_subtract), axis=0).astype(
            np.float64
        )
    else:
        mean_blurred_frame = np.zeros(
            [frames_for_bg_subtract.shape[1], frames_for_bg_subtract.shape[2]],
            dtype=np.float32,
        )
        max_blurred_frame = np.zeros(
            [frames_for_bg_subtract.shape[1], frames_for_bg_subtract.shape[2]],
            dtype=np.float32,
        )
        if use_multithreading:
            blurred_frames = run_with_threads(
                lambda i: cv2.GaussianBlur(frames_for_bg_subtract[i], (5, 5), 0),
                list(range(frames_for_bg_subtract.shape[0])),
                max_workers=max_workers,
            )
            for blurred in blurred_frames:
                mean_blurred_frame += blurred
                max_blurred_frame = np.maximum(max_blurred_frame, np.abs(blurred))
        else:
            for i in range(frames_for_bg_subtract.shape[0]):
                blurred = cv2.GaussianBlur(frames_for_bg_subtract[i], (5, 5), 0)
                mean_blurred_frame += blurred
                max_blurred_frame = np.maximum(max_blurred_frame, np.abs(blurred))

        mean_blurred_frame /= frames_for_bg_subtract.shape[0]
    max_blurred_frame -= mean_blurred_frame
    mean_normalization_value = np.max(max_blurred_frame)

    return mean_blurred_frame, mean_normalization_value


def compute_echogram(
    unwarped_frames,
    mean_blurred_frame=None,
    mean_normalization_value=None,
    return_echogram_with_bg_subtracted=True,
    return_echogram_with_how_wide_the_peak_as_third_channel=False,
    return_echogram_with_no_bgs_as_third_channel=False,
    return_echogram_with_distances_as_third_channel=False,
):
    """
    Generate an echogram from unwarped beam frames.

    Output channels:
    0: magnitude (max over bins)
    1: normalized argmax bin index in [-0.5, 0.5)
    2: (optional) magnitude without bg subtraction, peak width, or distances placeholder
    """
    assert (
        return_echogram_with_how_wide_the_peak_as_third_channel
        + return_echogram_with_no_bgs_as_third_channel
        + return_echogram_with_distances_as_third_channel
        <= 1
    ), "Cannot have more than 3 channels"

    if (
        return_echogram_with_distances_as_third_channel
        or return_echogram_with_no_bgs_as_third_channel
        or return_echogram_with_how_wide_the_peak_as_third_channel
    ):
        output = np.zeros(
            (unwarped_frames.shape[0], unwarped_frames.shape[1], 3),
            dtype=np.float32,
        )
    else:
        output = np.zeros(
            (unwarped_frames.shape[0], unwarped_frames.shape[1], 2),
            dtype=np.float32,
        )

    frames_f32 = unwarped_frames.astype(np.float32)

    no_bgs_echogram = None
    if return_echogram_with_no_bgs_as_third_channel:
        no_bgs_echogram = np.max(frames_f32, axis=2) / 255.0

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

    if return_echogram_with_how_wide_the_peak_as_third_channel:
        peak_vals = output[:, :, 0].astype(np.float32)
        peak_idx = angle_echogram
        thr = 0.25 * peak_vals
        above = proc >= thr[..., None]

        h, w, d = proc.shape
        width = np.zeros((h, w), dtype=np.float32)

        for r in range(h):
            above_r = above[r]
            peak_r = peak_idx[r]
            peakv_r = peak_vals[r]

            for c in range(w):
                pv = float(peakv_r[c])
                if not np.isfinite(pv) or pv <= 0.0:
                    width[r, c] = 0.0
                    continue

                p = int(peak_r[c])
                l = p
                while l > 0 and above_r[c, l - 1]:
                    l -= 1
                rr = p
                while rr < d - 1 and above_r[c, rr + 1]:
                    rr += 1
                width[r, c] = float(rr - l + 1)

        output[:, :, 2] = width

    if return_echogram_with_no_bgs_as_third_channel:
        output[:, :, 2] = no_bgs_echogram.astype(np.float32)

    if return_echogram_with_distances_as_third_channel:
        pass  # TODO: add distances echogram

    return output

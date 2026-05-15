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

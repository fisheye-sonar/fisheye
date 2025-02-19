import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import zoom

from fisheye.dataloaders.aris import create_aris_dataloader
from utils.generate_echograms import make_echogram_image, zero_pad_to_match_one_dim


def make_vid_from_np_stack(output_filename, image_stack, frame_rate=12, norm=False):
    """
    Save a stack of numpy images to an MP4 video file.

    :param image_stack: NumPy array of shape (num_frames, height, width, channels).
    :param output_filename: Name of the output MP4 file.
    :param fps: Frames per second of the output video.
    """

    if isinstance(image_stack, list):
        image_stack = np.stack(image_stack)

    if norm:
        image_stack = image_stack.astype(float)
        image_stack -= np.min(image_stack)
        image_stack /= np.max(image_stack)

    if np.max(image_stack) <= 1:
        scale_factor = 255
    else:
        scale_factor = 1

    num_frames, height, width, channels = image_stack.shape

    # Ensure images are in uint8 format
    # if image_stack.dtype != np.uint8:
    image_stack = (scale_factor * image_stack).astype(np.uint8)

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v' for MP4
    if "mp4" in output_filename:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")

    else:
        fourcc = cv2.VideoWriter_fourcc(*"X264")

    out = cv2.VideoWriter(output_filename, fourcc, frame_rate, (width, height))

    # for _ in range(fps * duration):
    #     frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    #     # Convert RGB (NumPy default) to BGR (OpenCV default)

    for frame in image_stack:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print(f"Video saved as {output_filename}")


def make_gif_from_np_stack(fn, frames, frame_rate=25, norm=False):
    """
    WARNING:RARE COLOURS ARE REMOVED IN THE COLOUR QUANTIZATION
    """
    if isinstance(frames, list):
        frames = np.stack(frames)

    if norm:
        frames = frames.astype(float)
        frames -= np.min(frames)
        frames /= np.max(frames)

    if np.max(frames) <= 1:
        scale_factor = 255
    else:
        scale_factor = 1

    images = [img * scale_factor for img in frames]
    images = [np.uint8(img) for img in images]
    pil_images = [Image.fromarray(img) for img in images]
    pil_images[0].save(
        fn, save_all=True, append_images=pil_images[1:], duration=1 / frame_rate, loop=0
    )
    print(f"GIF saved as {fn}")


def generate_echogram_gif_from_aris(
    config,
    save_filename,
    echogram_pop,
    return_unwarped,
    resize_mode="pad",
    return_list=False,
):

    vis = generate_echogram_vis_from_aris(
        config,
        echogram_pop,
        return_unwarped,
        resize_mode,
        return_list,
    )

    if save_filename:
        print("Saving gif...")
        make_gif_from_np_stack(save_filename, vis, frame_rate=25)


def generate_echogram_vis_from_aris(
    config,
    echogram_pop,
    return_unwarped,
    resize_mode="scale",
    return_list=False,
    colour_image_edges=True,
):
    dataloader, dataset = create_aris_dataloader(config)

    echograms = []
    frames_vis = []
    for i, batch in enumerate(dataloader):
        frames, echogram = (
            batch[0],
            batch[3],
        )
        if frames.shape[0] != 0:
            frames_vis.append(frames)
            echograms.append(echogram)
    echograms = np.concatenate(echograms, axis=0)
    frames_vis = np.concatenate(frames_vis, axis=0)

    coloured_echogram = make_echogram_image(
        echograms.astype("float"),
        echogram_pop=echogram_pop,
        filter_kernel=config.echogram_filter_kernel,
        filter_tol=config.echogram_filter_tol,
    )
    coloured_echogram = coloured_echogram[: frames_vis.shape[0]]

    scale_factor = frames_vis.shape[1] / coloured_echogram.shape[1]

    # if its within 5% dont modify the pixels, just crop, this is because the scaling
    if coloured_echogram.shape[1] < frames_vis.shape[1]:
        # pad
        coloured_echogram = zero_pad_to_match_one_dim(
            coloured_echogram, frames_vis.shape, dim=1, centered=True, const_value=125
        )
    elif coloured_echogram.shape[1] > frames_vis.shape[1] and scale_factor > 0.95:
        # crop
        height_diff = coloured_echogram.shape[1] - frames_vis.shape[1]
        coloured_echogram = coloured_echogram[
            :, int(height_diff / 2) : int(height_diff / 2) + frames_vis.shape[1]
        ]
    else:
        if resize_mode == "scale":

            scale_factor = frames_vis.shape[1] / coloured_echogram.shape[1]
            coloured_echogram = zoom(
                coloured_echogram, (1, scale_factor, 1, 1), order=1
            )  # order=1 for bilinear interpolation

            # can still be off by a few from a rounding error
            if coloured_echogram.shape[1] < frames_vis.shape[1]:
                coloured_echogram = zero_pad_to_match_one_dim(
                    coloured_echogram, frames_vis.shape, dim=1, const_value=125
                )
            elif coloured_echogram.shape[1] > frames_vis.shape[1]:
                coloured_echogram = coloured_echogram[:, : frames_vis.shape[1]]

        elif resize_mode == "pad":
            coloured_echogram = zero_pad_to_match_one_dim(
                coloured_echogram, frames_vis.shape, dim=1, const_value=125
            )
            frames_vis = zero_pad_to_match_one_dim(
                frames_vis,
                coloured_echogram.shape,
                dim=1,
                centered=False,
                const_value=125,
            )
    frames_vis = np.stack([frames_vis[:, :, :, 0]] * 3, axis=-1)
    if colour_image_edges:
        frames_vis_mask = np.max(frames_vis, axis=0)
        frames_vis_mask = np.where(frames_vis_mask > 0, 0, 1).astype(np.uint8)
        frames_vis_mask_col = frames_vis_mask.copy()
        frames_vis_mask_col[
            -int(frames_vis_mask_col.shape[0] / 2) :,
            :10,
            0,
        ] *= 150
        frames_vis_mask_col[
            -int(frames_vis_mask_col.shape[0] / 2) :,
            -10:,
            2,
        ] *= 150
        frames_vis_mask_col[
            -int(frames_vis_mask_col.shape[0] / 2) :,
            -10:,
            1,
        ] *= 50
        frames_vis = frames_vis + frames_vis_mask_col
    if coloured_echogram.shape[2] < frames_vis.shape[2]:
        coloured_echogram = np.repeat(
            coloured_echogram,
            int(frames_vis.shape[2] / coloured_echogram.shape[2]),
            axis=2,
        )
    comb = np.concatenate([frames_vis, coloured_echogram], axis=2)
    if return_list:
        return [x for x in comb]
    else:
        return comb

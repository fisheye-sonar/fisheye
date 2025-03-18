import argparse
import glob
import os
import subprocess as sp
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

from fisheye.dataloaders.didson.pyDIDSON import DIDSON
from fisheye.scripts.load_predictions import load_mot_file


# def load_predictions(root_dir):
#     dates = os.listdir(root_dir)
#
#     df_list = []
#     text_files = glob.glob(f"{root_dir}/*.txt")
#
#     if text_files:
#         non_empty_files = [f for f in text_files if os.path.getsize(f) > 0]
#
#         tmp_df = pd.concat(
#             (pd.read_csv(f, delimiter=",", header=None).assign(filename=os.path.basename(f), path=f) for f in
#              non_empty_files),
#             ignore_index=True
#         )
#         df_list.append(tmp_df)
#
#     df_detections = pd.concat(df_list, ignore_index=True)
#
#     column_names = ["frame_id", "fish_id", "right", "top", "width", "height", "conf", "filename", "filepath"]
#     df_detections.columns = column_names
#
#     return df_detections


def draw_bbox_on_frame(frame, bboxes):
    """
    Draw bounding boxes on an image frame.

    Args:
        frame (numpy array or PIL.Image): Image frame data.
        bboxes (list): List of bounding box dictionaries with keys 'right', 'top', 'width', and 'height'.

    Returns:
        Image: PIL Image with bounding boxes drawn.
    """
    # Convert frame to PIL Image if not already
    image = Image.fromarray(frame) if not isinstance(frame, Image.Image) else frame
    font = ImageFont.load_default(
        size=30
    )  # Use default font (can be replaced with a TrueType font)
    if image.mode != "L":
        image = image.convert("L")  # Ensure the image is grayscale

    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    for bbox in bboxes:
        fish_id = bbox["fish_id"]
        left_pixel = image_width * bbox["right"]
        top_pixel = image_height * bbox["top"]
        right_pixel = left_pixel + (bbox["width"] * image_width)
        bottom_pixel = top_pixel + (bbox["height"] * image_height)

        # Outer thicker box (White)
        draw.rectangle(
            [(left_pixel - 3, top_pixel - 3), (right_pixel + 3, bottom_pixel + 3)],
            outline=(255, 255, 255),
            width=7,
        )

        # Inner thinner box (Red)
        draw.rectangle(
            [(left_pixel, top_pixel), (right_pixel, bottom_pixel)],
            outline=(255, 0, 0),
            width=2,
        )

        # Draw the fish ID above the bounding boxes
        fish_id_position = (left_pixel, max(top_pixel - 40, 0))
        draw.text(fish_id_position, str(fish_id), fill=(255, 255, 255), font=font)

    return image


def process_frames_with_bboxes(df, frames, start_frame_id=0, end_frame_id=None):
    """
    Process frames and add bounding boxes where they exist, ensuring all frames are included in order.

    Args:
        df (pd.DataFrame): DataFrame containing detection information.
        frames (dict): Dictionary of frame_id to frame data (numpy arrays or bytes).

    Returns:
        List: List of PIL Image objects representing the frames with bounding boxes.
    """
    modified_frames = []

    # Determine the range of frames to process
    end_frame_index = (
        len(frames) if end_frame_id is None else end_frame_id - start_frame_id + 1
    )
    frames_to_process = frames[:end_frame_index] if end_frame_id is not None else frames

    for index, frame_data in enumerate(frames):
        frame_id = start_frame_id + index  # Calculate the actual frame ID

        # Check if there are bounding boxes for this frame
        if frame_id in df["frame_id"].values:
            # Get bounding boxes for the frame
            bboxes = df[df["frame_id"] == frame_id][
                ["right", "top", "width", "height", "fish_id"]
            ].to_dict(orient="records")

        else:
            # No bounding boxes for this frame
            bboxes = []

        # Draw bounding boxes (if any) on the frame
        modified_frame = draw_bbox_on_frame(frame_data, bboxes)
        modified_frames.append(modified_frame)

    return modified_frames


def make_video(out_file, frames, fps=20, do_bg_subtraction=True, cmap=False):
    command = [
        "ffmpeg",
        "-y",  # (optional) overwrite output file if it exists
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-r",
        str(fps),  # Frames per second
        "-i",
        "-",  # Input comes from a pipe
        "-an",  # No audio
        "-vcodec",
        "mpeg4",
        "-b:v",
        "10000k",
        out_file,  # Output file path
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    with sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE) as pipe:
        buffer = io.BytesIO()
        for im in frames:
            if not isinstance(im, Image.Image):
                im = Image.fromarray(im)

            buffer.seek(0)
            buffer.truncate()
            im.save(buffer, "JPEG")
            buffer.seek(0)  # Move to the beginning of the buffer

            try:
                pipe.stdin.write(buffer.read())
            except BrokenPipeError:
                print("Broken pipe error while writing frame to FFMPEG.")
                break
            finally:
                buffer.seek(0)
                buffer.truncate()  # Clear the buffer for the next frame

        pipe.stdin.close()
        pipe.wait()

        # Capture stderr for debugging
        stderr_output = pipe.stderr.read().decode()
        if stderr_output:
            print(f"FFMPEG error: {stderr_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="ARIS/DIDSON filepath", required=True)
    parser.add_argument(
        "--mot_dir", type=str, help="Path to MOT txt file(s)", required=True
    )
    parser.add_argument(
        "--out_dir", type=str, help="Path to save file(s) to.", required=True
    )

    args = parser.parse_args()

    # Load predictions
    df = load_mot_file(args.mot_dir)

    file_stem = Path(args.file).stem

    # Load frames
    didson = DIDSON(args.file)
    frames, _ = didson.load_frames()
    print(f"Loaded {len(frames)} frames")

    # Filter MOT results to detections for this file
    modified_frames = process_frames_with_bboxes(
        df[df["file_stem"] == file_stem + ".txt"], frames
    )
    print("Creating video...")
    make_video(out_file=f"{args.out_dir}/{file_stem}.mp4", frames=modified_frames)

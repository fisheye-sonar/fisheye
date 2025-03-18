import argparse
import io
import os
import subprocess as sp

import numpy as np
from PIL import Image

from fisheye.dataloaders.didson.pyDIDSON import DIDSON


def convert_to_mp4(frames, destination, fps=12):
    """Convert ARIS/DIDSON file to MP4 files.

    Uses FFMPEG to generate a mp4 video from NumPy arrays
    """

    num_frames, height, width = frames.shape
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
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
        "mjpeg",
        "-b:v",
        "10000k",  # bitrate
        destination,  # Output file path
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    try:
        # Open the ffmpeg process
        with sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE) as pipe:
            # Stream frames to ffmpeg one by one
            for i, frame in enumerate(frames):
                try:
                    image = Image.fromarray(frame).convert("RGB")

                    # Encode as JPEG
                    with io.BytesIO() as img_buffer:
                        image.save(img_buffer, format="JPEG", quality=95)
                        jpeg_bytes = img_buffer.getvalue()

                    # Write JPEG bytes to FFmpeg process
                    pipe.stdin.write(jpeg_bytes)

                    # Flush after each frame to ensure it's sent to ffmpeg
                    pipe.stdin.flush()

                except Exception as e:
                    print(f"Error while writing frame {i + 1} to ffmpeg: {e}")
                    break  # Exit the loop if we encounter an issue

            # Close stdin and wait for ffmpeg process to finish
            pipe.stdin.close()
            pipe.wait()

            # Capture stderr for debugging
            stderr_output = pipe.stderr.read().decode()
            if stderr_output:
                print(f"FFMPEG error: {stderr_output}")

            del frames

    except BrokenPipeError:
        print("Broken pipe error occurred during FFMPEG processing.")
        pipe.terminate()
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        pipe.terminate()
        raise


def convert_to_jpeg(file_path, output_dir, format, fps):
    """Convert ARIS/DIDSON file to JPEGs."""
    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)

    print(f"Loading frames from {file_path}")
    didson = DIDSON(file=file_path)
    print(
        f"Start frame: {didson.info['startframe']}\n End frame: {didson.info['endframe']}"
    )
    frames, _ = didson.load_frames()

    if format == "mp4":
        convert_to_mp4(frames, os.path.join(file_output_dir, f"{filename}.mp4"), fps)

    else:
        for idx, frame in enumerate(frames):
            image_name = f"{idx}.jpg"
            image_path = os.path.join(file_output_dir, image_name)
            Image.fromarray(frame).save(image_path)

        print(f"Frames saved to: {file_output_dir}")


def process_file(input_path, output_dir, format, fps):
    """Convert ARIS/DIDSON file to either JPEG or MP4."""
    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.endswith((".ddf", ".aris")):
                convert_to_jpeg(os.path.join(input_path, file), output_dir, format, fps)
    elif os.path.isfile(input_path) and input_path.endswith((".ddf", ".aris")):
        convert_to_jpeg(input_path, output_dir, format, fps)
    else:
        print(f"Invalid input: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="Local path to ARIS/DIDSON file(s).", required=True
    )
    parser.add_argument(
        "--output", type=str, help="Local path to save file(s) to.", required=True
    )
    parser.add_argument(
        "--format", type=str, choices=["jpeg", "mp4"], default="jpeg", required=True
    )
    parser.add_argument("--fps", type=int, default=12, required=False)
    args = parser.parse_args()

    process_file(args.input, args.output, args.format, args.fps)

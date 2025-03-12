import ast
import gc
import io
import logging
import os
import tempfile
import subprocess as sp
import time

import numpy as np
import pandas as pd
from PIL import Image
import boto3
from botocore.exceptions import ClientError
import torch

from fisheye.configs import ARISDatasetConfig
from fisheye.dataloaders import ARISBatchedDataset
from utils.visualisation_utils import generate_echogram_vis_from_aris


logging.basicConfig(
    filename='vancouver_island_failures.log',
    filemode="a",  # Append mode
    #format='%(asctime)s - File: %(filename)s - Frame: %(frame)s - Error: %(message)s',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)


# logging.basicConfig(
#     level=logging.ERROR,
#     format="%(asctime)s - File: %(filename)s - Frame: %(frame)s - Error: %(message)s",
#     handlers=[logging.FileHandler("clearwater_error.log"), logging.StreamHandler()],
# )
# logger = logging.getLogger(__name__)
# console_handler = logging.StreamHandler()
# file_handler = logging.FileHandler("clearwater_failures_mvh.log", mode="a", encoding="utf-8")
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)
# formatter = logging.Formatter(
#     "{asctime} - {levelname} - {message}",
#      style="{",
#      datefmt="%Y-%m-%d %H:%M",
#  )

# console_handler.setFormatter(formatter)

# # Create handlers
# console_handler = logging.StreamHandler()
# # file_handler = logging.FileHandler("error.log")

# # Set level for handlers
# console_handler.setLevel(logging.ERROR)
# # file_handler.setLevel(logging.ERROR)

# # Create formatter and add it to handlers
# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# console_handler.setFormatter(formatter)
# # file_handler.setFormatter(formatter)


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.ERROR)
# # Add handlers to the logger
# logger.addHandler(console_handler)
# # logger.addHandler(file_handler)

BUCKET = "fishcounting"
s3 = boto3.client("s3", region_name="us-east-2")

dir_path = "/datadrive"
data_drive_plus_dirs = [
    "ARIS_2024_10_28",
    "ARIS_2024_10_29",
    "ARIS_2024_10_30",
    "ARIS_2024_10_31",
]
# csv_fp = "2025-01-22_klamath_202410_annotation_batch_0_merged.csv"

# Load csv to get frame ranges
# data = pd.read_csv(os.path.join(dir_path, csv_fp))
# data = pd.read_csv('/Users/madison/Downloads/2025-01-22_klamath_202410_annotation_batch_0_merged.csv')
# data = pd.read_csv("/home/mahobley/Code/fisheye/2024_klamath_last_15.csv")
data = pd.read_csv(
    "/home/mahobley/Code/fisheye/2024_seattle_harddrives_EASY_MEDIUM_HARD_results.csv"
)
# hard_drive = "klamath"
data["frame_ranges"] = data["frame_ranges"].apply(ast.literal_eval)
data["frame_diff"] = data["frame_ranges"].apply(lambda x: x[1] - x[0])
# Sort by the 'frame_diff' column
data = data.sort_values(by="frame_diff", ascending=True)

# Drop the 'frame_diff' column if you no longer need it
data = data.drop(columns=["frame_diff"])
print(data.columns)

fps = 12
echogram_filter_kernel = 7
echogram_filter_tol = 0.15
return_unwarped = False
return_echogram = True


def generate_video(
    frames: np.ndarray, output_fp: str = None, fps: float = 20.0
) -> dict:
    """Uses FFMPEG to generate a lossless mp4 video from NumPy arrays.

    Writes mp4 to temporary file.
    """

    num_frames, height, width, _ = frames.shape
    is_color = frames.ndim == 4
    pixel_format = "gray" if not is_color else "rgb24"

    frames = np.ascontiguousarray(frames, dtype=np.uint8)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_file:
        temp_file_path = output_fp or temp_file.name

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
            "10000k",
            temp_file_path,  # Output file path
            "-hide_banner",
            "-loglevel",
            "error",
        ]

        try:
            # Open the ffmpeg process
            with sp.Popen(
                command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE
            ) as pipe:
                # Stream frames to ffmpeg one by one
                for i, frame in enumerate(frames):
                    try:
                        image = Image.fromarray(frame)

                        # Encode as JPEG
                        with io.BytesIO() as img_buffer:
                            image.save(img_buffer, format="JPEG", quality=95)
                            jpeg_bytes = img_buffer.getvalue()

                        # Write JPEG bytes to FFmpeg process
                        pipe.stdin.write(jpeg_bytes)

                        # Flush after each frame to ensure it's sent to ffmpeg
                        pipe.stdin.flush()

                        # Log for debugging
                        logger.debug(f"Written frame {i + 1} to stdin.")

                    except Exception as e:
                        logger.error(
                            f"Error while writing frame {i + 1} to ffmpeg: {e}"
                        )
                        break  # Exit the loop if we encounter an issue

                # Close stdin and wait for ffmpeg process to finish
                pipe.stdin.close()
                pipe.wait()

                # Capture stderr for debugging
                stderr_output = pipe.stderr.read().decode()
                if stderr_output:
                    logger.error(f"FFMPEG error: {stderr_output}")

        except BrokenPipeError:
            logger.error("Broken pipe error occurred during FFMPEG processing.")
            pipe.terminate()
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            pipe.terminate()
            raise

        with open(temp_file_path, "rb") as f:
            content = f.read()

    return {"filename": "asset.mp4", "content": content}


def merge_overlapping_ranges_to_dict(df: pd.DataFrame) -> dict:
    def merge_ranges(frame_ranges):
        """Merge overlapping or adjacent frame ranges."""
        intervals = sorted(frame_ranges)
        merged = []

        for start, end in intervals:
            # No overlap
            if not merged or merged[-1][1] < start:
                merged.append((start, end))
            else:  # Overlap, merge intervals
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        return merged

    grouped = df.groupby(["hard_drive", "filename"])["frame_ranges"].apply(list)

    merged_dict = {}
    for (hard_drive, filename), frame_ranges in grouped.items():
        merged_dict[(hard_drive, filename)] = merge_ranges(frame_ranges)

    return merged_dict


def try_generate_frames(config, filename, start_frame_idx, end_frame_idx, retry=True):
    """Attempts to generate frames. Retries with a reduced end_frame_idx if enabled."""
    try:
        return generate_echogram_vis_from_aris(
            config,
            echogram_pop=True,
            return_unwarped=return_unwarped,
            resize_mode="scale",
            return_list=False,
            colour_image_edges=True,
            echogram_filter_kernel=echogram_filter_kernel,  # Assuming defaults
            echogram_filter_tol=echogram_filter_tol,  # Assuming defaults
        )
    except Exception as e:
        logging.error(
            "Failed to load frames. mot_filename: %s, frame_range: %s, error: %s",
            filename,
            [start_frame_idx, end_frame_idx],
            str(e),
        )

        gc.collect()

        if retry:
            config.end_frame = -1  # Update config
            tmp_dataset = ARISBatchedDataset(config)
            config.end_frame = tmp_dataset.end_frame
            print(f'New end frame value from retry block: {config.end_frame}')
            del tmp_dataset


            try:
                return generate_echogram_vis_from_aris(
                    config,
                    echogram_pop=True,
                    return_unwarped=False,
                    resize_mode="scale",
                    return_list=False,
                    colour_image_edges=True,
                    echogram_filter_kernel=echogram_filter_kernel,
                    echogram_filter_tol=echogram_filter_tol,
                )
            except Exception as e:
                logger.error(
                    "Retry failed. Skipping frame range. "
                    "mot_filename: %s, frame_range: %s, error: %s",
                    filename,
                    [start_frame_idx, end_frame_idx],
                    str(e),
                )

                return None  # Skip this frame range


def s3_file_exists(bucket_name, object_key):
    """Check whether an object exists in an S3 bucket."""
    try:
        s3.head_object(Bucket=bucket_name, Key=object_key)
        return True  # File exists
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False  # File does not exist
        else:
            raise  # Some other error occurred


def list_s3_files(bucket_name, prefix=""):
    """Get a list of all files in an S3 bucket with pagination."""
    files = []
    continuation_token = None

    while True:
        list_params = {"Bucket": bucket_name, "Prefix": prefix}
        if continuation_token:
            list_params["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**list_params)

        if "Contents" in response:
            files.extend([obj["Key"] for obj in response["Contents"]])

        if response.get("IsTruncated"):  # More files to fetch
            continuation_token = response["NextContinuationToken"]
        else:
            break

    return files


def download_s3_file(bucket_name, s3_key, local_path):
    """Downloads file from s3 to local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)  # Ensure directory exists
    s3.download_file(bucket_name, s3_key, local_path)
    print(f"Downloaded {s3_key} to {local_path}")


def split_frame_ranges(data, threshold=1000):
    def split_ranges(ranges):
        new_ranges = []
        for start, end in ranges:
            while start + threshold < end:
                new_ranges.append((start, start + threshold))
                start += threshold
            new_ranges.append((start, end))
        return new_ranges

    if isinstance(data, dict):
        return {key: split_ranges(ranges) for key, ranges in data.items()}

    elif isinstance(data, pd.DataFrame):
        expanded_rows = []
        for _, row in data.iterrows():
            split_ranges_list = split_ranges([row["frame_range"]])
            for new_range in split_ranges_list:
                expanded_rows.append(
                    {"filename": row["filename"], "frame_range": new_range}
                )
        return pd.DataFrame(expanded_rows)

    else:
        raise TypeError("Input data must be a dictionary or a pandas DataFrame")


prefix = "annotation_staging_area/vancouver-island-batch-0/"
# s3_data_subdir = f"data/klamath_data"
curr_s3_files = list_s3_files(BUCKET, prefix)
print(f'{len(curr_s3_files)} files in s3')
# Local path to data
data_path = "/home/mahobley/Code/fisheye/madi_data_todel/"

merged_dict = merge_overlapping_ranges_to_dict(data[data['hard_drive'] == '2024_06_Seattle_Fish_Counting_VancouverIsland'])
split_range_dict = split_frame_ranges(merged_dict)

# merged_dict = merge_overlapping_ranges_to_dict(
#     data[data["hard_drive"] == "2024_06_Seattle_Fish_Counting_Clearwater"]
# )
for key, value in split_range_dict.items():
    hard_drive, filename = key
    for frame_range in value:
        # for row in data.itertuples(index=False):
        #     filename = row.mot_file
        #     subdir = row.subdir
        #     frame_range = row.frame_range
        # if subdir in data_drive_plus_dirs:
        #     data_path = f'/datadrive3/2024_klamath_data_datadriveplus/{subdir}'
        #
        # else:
        #     data_path = f'/datadrive/2024_klamath_data/{subdir}'

        print(f"Processing {hard_drive}: {filename}")

        # if f"{prefix}/{filename[:-4]}.mp4" in s3_files:
        #     print(f"Skipping {filename}")
        #     continue

        # if filename =='HARD_2023-10-19-Nanaimo_RightBank_2023-10-19-170000_2060_4120.txt':
        #     continue
        # HARD_2022-10-31_Nanaimo_RightBank_2022-10-31-170000

        aris_file_name = filename[:-4] + ".aris"
        fp = os.path.join(data_path, aris_file_name)

        # if f"{prefix}/{filename[:-4]}.mp4" in s3_files:
        #     print(f"Skipping {filename}")
        #     continue

        # if not os.path.exists(fp):
        #     print(f"{fp} not found locally, downloading from S3...")
        #     tmp_subdir = f"{subdir}/{aris_file_name}"
        #     download_s3_file(BUCKET, f"{s3_data_subdir}/{tmp_subdir}", fp)

        # tmp_dataset = ARISBatchedDataset(
        #     ARISDatasetConfig(
        #         filepath=fp,
        #         start_frame=frame_range[0],
        #         end_frame=0,
        #         batch_size=1,
        #         return_unwarped=return_unwarped,
        #         return_echogram=return_echogram,
        #     )
        # )
        # end_frame = tmp_dataset.end_frame

        # del tmp_dataset

        start_frame_idx = frame_range[0]
        # end_frame_idx = min(frame_range[1] + 1, end_frame+1)   # Make sure end frame index is inclusive
        end_frame_idx = frame_range[1] +1

        tmp_file = f"{prefix}{filename[:-4]}_{start_frame_idx}_{end_frame_idx-1}.mp4"
        print(f'Potential file name in s3: {tmp_file}')
        
        # if end_frame_idx - start_frame_idx > 1000:
        #     print(f'Skipping frame range - too large. {filename}, {start_frame_idx}-{end_frame_idx}')
        #     logger.error(
        #         "Skipping frame range - too large. "
        #         "mot_filename: %s, frame_range: %s",
        #         filename,
        #         [start_frame_idx, end_frame_idx],
        #     )
        #     continue

        if tmp_file in curr_s3_files:
            print(f"Skipping {aris_file_name}. Already in S3. \n")
            continue

        if not os.path.exists(fp):
            print(f"{fp} not found locally, downloading from S3...")
            # subdir = f"seattle_conference_data/2024_06_Seattle_Fish_Counting_Dungeness_LeftBank_Data/videos"
            #subdir = f"seattle_conference_data/Dungeness_Clearwater_not_encrypted/2024_06_Seattle_Fish_Counting_[Clearwater]/videos"
            subdir = f"seattle_conference_data/Vancouver_Island/{hard_drive}/Videos"
            tmp_subdir = f"{subdir}/{aris_file_name}"
            # print(f"{s3_data_subdir}/{tmp_subdir}")
            # download_s3_file(BUCKET, f"{s3_data_subdir}/{tmp_subdir}
            print(f'S3 prefix for pulling file locally: {tmp_subdir}')
            download_s3_file(BUCKET, f"{tmp_subdir}", fp)

        print(f"Current frame range for {filename}: {start_frame_idx}-{end_frame_idx}")
        # mp4_file = f"annotation_staging_area/{hard_drive}-batch-0/{aris_file_name[:-5]}_{start_frame_idx}_{end_frame_idx-1}.mp4"
        # print(f'Potential MP4 name: {mp4_file}')
        # if s3_file_exists(BUCKET, mp4_file):
        #   print(f"MP4 file exists in S3: {mp4_file}")
        #  continue

        start = time.time()
        max_frame_idx = end_frame_idx

        config = ARISDatasetConfig(
            filepath=fp,
            start_frame=start_frame_idx,
            end_frame=end_frame_idx,
            batch_size=1,
            return_unwarped=return_unwarped,
            return_echogram=return_echogram,
        )
        print("Running try_generate_frames()")
        frames = try_generate_frames(config, filename, start_frame_idx, end_frame_idx)

        if frames is None:
            continue  # Skip this frame range and move to the next one

        if len(frames) < end_frame_idx:
            end_frame_idx = len(frames) + start_frame_idx

        print("Running generate_video()")
        output = generate_video(frames=frames, fps=fps)

        print("Put video in S3")
        s3.put_object(
            Bucket=BUCKET,
            Key=f"annotation_staging_area/vancouver-island-batch-0/"
            f"{aris_file_name[:-5]}_{start_frame_idx}_{end_frame_idx}.mp4",
            Body=output["content"],
        )

        print("Finished uploading to S3")

        end = time.time()
        print(f"Completed {aris_file_name} in {end - start:.2f} seconds")
        del frames
        torch.cuda.empty_cache()


# # for row in data.itertuples(index=False):
# #     filename = row.mot_file
# #     subdir = row.subdir
# #     frame_range = row.frame_range
#     # if subdir in data_drive_plus_dirs:
#     #     data_path = f'/datadrive3/2024_klamath_data_datadriveplus/{subdir}'
#     #
#     # else:
#     #     data_path = f'/datadrive/2024_klamath_data/{subdir}'

#     print(f"Processing {hard_drive}: {filename}")

#     # if f"{prefix}/{filename[:-4]}.mp4" in s3_files:
#     #     print(f"Skipping {filename}")
#     #     continue

#     aris_file_name = filename[:-4] + ".aris"
#     fp = os.path.join(data_path, aris_file_name)

#     # if f"{prefix}/{filename[:-4]}.mp4" in s3_files:
#     #     print(f"Skipping {filename}")
#     #     continue

#     # if not os.path.exists(fp):
#     #     print(f"{fp} not found locally, downloading from S3...")
#     #     tmp_subdir = f"{subdir}/{aris_file_name}"
#     #     download_s3_file(BUCKET, f"{s3_data_subdir}/{tmp_subdir}", fp)

#     print(f"Original start frame: {frame_range[0]}")
#     print(f"Original end frame: {frame_range[1]}")
#     start_frame_idx = frame_range[0]
#     end_frame_idx = frame_range[1] + 1  # Make sure end frame index is inclusive

#     tmp_file = f"{prefix}{filename[:-4]}_{start_frame_idx}_{end_frame_idx-1}.mp4"
#     print(f'Potential file name in s3: {tmp_file}')

#     if tmp_file in curr_s3_files:
#         print(f"Skipping {aris_file_name}")
#         continue

#     if not os.path.exists(fp):
#         print(f"{fp} not found locally, downloading from S3...")
#         subdir = f'{hard_drive}/Videos/'
#         tmp_subdir = f"{subdir}/{aris_file_name}"
#         download_s3_file(BUCKET, f"{s3_data_subdir}/{tmp_subdir}", fp)

#     print(f"Current frame range for {filename}: {start_frame_idx}-{end_frame_idx}")
#     # mp4_file = f"annotation_staging_area/{hard_drive}-batch-0/{aris_file_name[:-5]}_{start_frame_idx}_{end_frame_idx-1}.mp4"
#     # print(f'Potential MP4 name: {mp4_file}')
#     # if s3_file_exists(BUCKET, mp4_file):
#     #   print(f"MP4 file exists in S3: {mp4_file}")
#     #  continue

#     start = time.time()
#     max_frame_idx = end_frame_idx

#     config = ARISDatasetConfig(
#         filepath=fp,
#         start_frame=start_frame_idx,
#         end_frame=end_frame_idx,
#         batch_size=1,
#         return_unwarped=return_unwarped,
#         return_echogram=return_echogram,
#     )
#     print("Running try_generate_frames()")
#     frames = try_generate_frames(config, filename, start_frame_idx, end_frame_idx)

#     if frames is None:
#         continue  # Skip this frame range and move to the next one

#     if len(frames) < end_frame_idx:
#         end_frame_idx = len(frames) + start_frame_idx

#     print("Running generate_video()")
#     output = generate_video(frames=frames, fps=fps)

#     print("Put video in S3")
#     s3.put_object(
#         Bucket=BUCKET,
#         Key=f"annotation_staging_area/{hard_drive}-batch-0/"
#         f"{aris_file_name[:-5]}_{start_frame_idx}_{end_frame_idx}.mp4",
#         Body=output["content"],
#     )

#     print("Finished uploading to S3")

#     end = time.time()
#     print(f"Completed {aris_file_name} in {end - start:.2f} seconds")
#     del frames
#     torch.cuda.empty_cache()

import argparse

import boto3
import os

s3_client = boto3.client("s3")


def download_file(bucket, object_name, destination):
    """Download a file from S3 and save it locally."""
    s3_client.download_file(bucket, object_name, destination)
    print(f"Downloaded {object_name} to {destination}.")


def s3_download(bucket, prefix, destination_folder):
    """Download all files from an S3 folder (prefix).

    Args:
        bucket (str): S3 bucket.
        prefix (str): S3 prefix path.
        destination_folder (str): Local path to download file(s), excluding file name.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" in response:
        for obj in response["Contents"]:
            file_key = obj["Key"]
            file_name = os.path.basename(file_key)
            destination_path = os.path.join(destination_folder, file_name)

            if file_name:  # Avoid empty filenames (e.g., if prefix is a folder itself)
                s3_client.download_file(bucket, file_key, destination_path)
                print(f"Downloaded {file_key} to {destination_path}")
    else:
        print(f"No files found in {prefix}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, help="S3 bucket name", required=True)
    parser.add_argument("--prefix", type=str, help="S3 prefix", required=True)
    parser.add_argument(
        "--destination", type=str, help="Local path to save file(s) to", required=True
    )
    args = parser.parse_args()

    s3_download(args.bucket, args.prefix, args.destination)

import os

from PIL import Image

from fisheye.dataloaders.didson.pyDIDSON import DIDSON


def export_frames_to_disk(root_dir, output_dir, beam_width_dir):
    """
    Processes DIDSON/ARIS files by extracting raw frames and saving them as JPEG images.

    Function will run through directories looking for DIDSON/ARIS files. Once frames have been extracted,
    a new directory (name of DIDSON/ARIS file) will be created to store the raw frames.
    Args:
        root_dir (str): Path to the root directory containing subdirectories with DIDSON/ARIS files.
        output_dir (str): Path to the directory where frames will be saved.
        beam_width_dir (str): Path to the directory with beam width information.
    """
    os.makedirs(output_dir, exist_ok=True)
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".ddf") or file.endswith(".aris"):
                didson_file_path = os.path.join(subdir, file)
                didson_filename = os.path.splitext(file)[0]
                file_output_dir = os.path.join(output_dir, didson_filename)

                os.makedirs(file_output_dir, exist_ok=True)  # Create output folder

                print(f"Processing file: {didson_file_path}")
                didson = DIDSON(file=didson_file_path, beam_width_dir=beam_width_dir)
                print(
                    f"Start frame: {didson.info['startframe']}\n End frame: {didson.info['endframe']}"
                )
                frames, unwarped = didson.load_frames()

                for idx, frame in enumerate(frames):
                    image_name = f"{idx}.jpg"
                    image_path = os.path.join(file_output_dir, image_name)
                    Image.fromarray(frame).save(image_path)

                print(f"Frames saved to: {file_output_dir}")


root_dir = ""  # Path to the root directory containing subdirs with DIDSON files
output_dir = ""  # Path to save extracted frames
beam_width_dir = ""  # Path to beam width information

export_frames_to_disk(root_dir, output_dir, beam_width_dir)

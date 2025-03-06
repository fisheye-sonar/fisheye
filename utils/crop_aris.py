import fisheye.dataloaders.didson.pyARIS as pyARIS
import struct


def crop_clip(aris_path, out_fp, num_frames, verbose=True):
    """
    Crop and save an aris file based on the first *num_frames* frames. This will create a new ARIS file where the
    startframe, endframe and numframes in the file header are updated to the new aris file data.

    Args:
    aris_path: (str) The path to the aris file.
    out_fp: (str) The path to the output file. Include .aris extension.
    num_frames: (int) The number of frames to crop.
    verbose: (bool)
    """
    # load aris file and extract frame size
    ARIS_data, frame = pyARIS.DataImport(aris_path)
    FrameSize = ARIS_data.NumRawBeams * ARIS_data.SamplesPerChannel
    if verbose:
        print(
            "True Old", ARIS_data.FrameCount, ARIS_data.StartFrame, ARIS_data.EndFrame
        )

    # get byte index of the cutoff point
    frameoffset = 1024 + (num_frames * (1024 + (FrameSize)))

    # Read aris the bytes for the head and frames we want and cast to bytearray
    data = open(ARIS_data.filename, "rb")
    cropped = data.read(frameoffset)
    array = bytearray(cropped)

    old_frame_count = struct.unpack("I", array[4:8])[0]
    old_start_frame = struct.unpack("I", array[352:356])[0]
    old_end_frame = struct.unpack("I", array[356:360])[0]

    if verbose:
        print("old", old_frame_count, old_start_frame, old_end_frame)

    new_start_frame = 0
    new_end_frame = new_start_frame + num_frames

    # Set new values
    array[4:8] = bytearray(struct.pack("I", num_frames))
    array[352:356] = bytearray(struct.pack("I", new_start_frame))
    array[356:360] = bytearray(struct.pack("I", new_end_frame))

    if verbose:
        print("new", array[4:8], array[352:356], array[356:360])

    # Cast to bytes
    cropped = bytes(array)

    # Save new aris file
    with open(out_fp, "wb") as f:
        f.write(cropped)

    # Load file to check that the frame count, start frame and end frame makes sense
    if verbose:
        ARIS_data_2, frame = pyARIS.DataImport(out_fp)
        print(
            "check",
            ARIS_data_2.FrameCount,
            ARIS_data_2.StartFrame,
            ARIS_data_2.EndFrame,
        )


crop_clip(aris_path="", out_fp="", num_frames=10)

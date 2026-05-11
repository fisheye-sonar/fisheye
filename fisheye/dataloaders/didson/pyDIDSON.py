"""
Utilities to read and produce to-scale images from DIDSON and ARIS sonar files.

Portions of this code were adapted from SoundMetrics MATLAB code.

@author kulits
"""

__version__ = "b1.0.2"

import contextlib
import os
import struct
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import numpy as np
import pandas as pd

from . import pyARIS
from .pyDIDSON_format import *

BASE = Path(__file__).parent.parent.parent
BEAM_WIDTH_DIR = (BASE / "beam_widths").resolve()


def compute_resized_shape(original_shape, img_size, stride, pad=0):
    """Return a stride-aligned resized shape that preserves aspect ratio.

    Parameters
    ----------
    original_shape : tuple[int, int]
        Source image shape as ``(height, width)``.
    img_size : int
        Target size for the longest dimension before stride alignment.
    stride : int
        Output dimensions are rounded up to a multiple of this stride.
    pad : int, optional
        Extra stride units to add after scaling.

    Returns
    -------
    np.ndarray
        Resized ``(height, width)`` as integer multiples of ``stride``.
    """
    aspect_ratio = original_shape[0] / original_shape[1]
    shape = [1, 1 / aspect_ratio] if aspect_ratio > 1 else [aspect_ratio, 1]
    return np.ceil(np.array(shape) * img_size / stride + pad).astype(int) * stride


class DIDSON:

    def __init__(
        self,
        file,
        beam_width_dir=BEAM_WIDTH_DIR,
        ixsize=-1,
        img_load_size=None,  # loads bigger than the final image size and then 'downsamples' to it
        img_size=None,
        stride=64,
        return_original_image=False,
    ):
        """Initialize a reader and precompute the warp used for frame loading.

        Parameters
        ----------
        file : file-like object, string, or pathlib.Path
            DIDSON or ARIS file to inspect.
        beam_width_dir : string or pathlib.Path, optional
            Directory containing ARIS beam-width calibration CSV files.
        ixsize : int, optional
            Desired warped image width. For DIDSON this is exact; for ARIS it is
            approximate because the final bounds are recomputed from metric pixel size.
        img_load_size : int, optional
            Desired intermediate warped height used when precomputing the mapping.
            Larger values preserve more detail before any later downsampling.
        img_size : int, optional
            Final model-facing height. When provided, an area-based remapping is
            precomputed from the larger warp grid to this output size.
        stride : int, optional
            Model stride used when reporting the derived output size.
        return_original_image : bool, optional
            If ``True``, also keep the full-resolution warp mapping so callers can
            reconstruct the original warped image alongside resized outputs.
        """
        self.return_original_image = return_original_image
        info = self.read_header(file)
        self.info, self.write_rows, self.write_cols, self.read_i = (
            DIDSON.compute_image_metadata(info, beam_width_dir, ixsize, img_load_size)
        )

        big_ydim = self.info["ydim"]
        big_xdim = self.info["xdim"]

        if self.return_original_image:
            self.big_ydim = big_ydim
            self.big_xdim = big_xdim
            self.big_write_rows = self.write_rows
            self.big_write_cols = self.write_cols
            self.big_read_i = self.read_i

        if img_size is not None:
            # get size
            y, x = img_size, int(np.ceil(img_size / (big_ydim / big_xdim)))
            print(
                f"Given the load size ({big_ydim},{big_xdim}), the model input size {img_size} and the stride {stride}, images will be indexed to size ({y},{x})"
            )
            self.out_ydim = y
            self.out_xdim = x

            self._pix_area, self._count_area = self.precompute_area_like_pix_and_count(
                self.write_rows,
                self.write_cols,
                big_ydim,
                big_xdim,
                self.out_ydim,
                self.out_xdim,
            )
        else:
            self._pix_area, self._count_area = self.precompute_area_like_pix_and_count(
                self.write_rows,
                self.write_cols,
                self.info["ydim"],
                self.info["xdim"],
                self.info["ydim"],
                self.info["xdim"],
            )
            self.out_ydim = self.info["ydim"]
            self.out_xdim = self.info["xdim"]

    def read_header(self, file: Union[str, Path]):
        """Parse file-level and frame-level headers from a DIDSON or ARIS file.

        Parameters
        ----------
        file : file-like object, str, or pathlib.Path
            Path to the file or open file object.

        Returns
        -------
        info : dict
            Dictionary containing header fields.
        """
        if hasattr(file, "read"):
            file_ctx = contextlib.nullcontext(file)
            filename = getattr(file, "name", None)
            filename = os.path.abspath(filename)
        else:
            file = Path(file).expanduser().resolve()
            file_ctx = open(file, "rb")
            filename = str(file)

        if filename:
            filename = os.path.abspath(filename)

        with file_ctx as fid:
            assert fid.read(3) == b"DDF"
            version_id = fid.read(1)[0]
            fid.seek(0)

            info = {"pydidson_version": __version__}

            file_attributes, frame_attributes = {
                0: NotImplementedError,
                1: NotImplementedError,
                2: NotImplementedError,
                3: [file_attributes_3, frame_attributes_3],
                4: [file_attributes_4, frame_attributes_4],
                5: [file_attributes_5, frame_attributes_5],
            }[version_id]

            fileheaderformat = "=" + "".join(file_attributes.values())
            fileheadersize = struct.calcsize(fileheaderformat)
            info.update(
                dict(
                    zip(
                        file_attributes.keys(),
                        struct.unpack(fileheaderformat, fid.read(fileheadersize)),
                    )
                )
            )

            frameheaderformat = "=" + "".join(frame_attributes.values())
            frameheadersize = struct.calcsize(frameheaderformat)
            info.update(
                dict(
                    zip(
                        frame_attributes.keys(),
                        struct.unpack(frameheaderformat, fid.read(frameheadersize)),
                    )
                )
            )

            info.update(
                {
                    "fileheaderformat": fileheaderformat,
                    "fileheadersize": fileheadersize,
                    "frameheaderformat": frameheaderformat,
                    "frameheadersize": frameheadersize,
                }
            )

            file_size = os.path.getsize(filename)
            framesize = info["samplesperchannel"] * info["numbeams"]
            numframes = int(
                np.floor((file_size - fileheadersize) / (frameheadersize + framesize))
            )

            info.update(
                {
                    "numframes": numframes,
                    "framesize": framesize,
                    "filename": filename,
                    "version_id": version_id,
                }
            )

            return info

    @staticmethod
    def compute_image_metadata(
        info: dict,
        beam_width_dir: Path = BEAM_WIDTH_DIR,
        ixsize: int = -1,
        img_load_size: int = 2688,  # our default image size is 896, this is 3x that for a less noisey final image
    ):
        """Derive warp metadata and sample-to-image mappings from parsed headers.

        Parameters
        ----------
        info : dict
            Parsed header information.
        beam_width_dir : Path
            Directory containing beam-width CSV files used by ARIS processing.
        ixsize : int
            Requested warped image width. ``-1`` keeps the default for the file type.
        img_load_size : int
            Requested warped image height for ARIS remapping. Ignored for DIDSON.

        Returns
        -------
        info : dict
            Header dictionary augmented with derived geometry and pixel-size metadata.
        write_rows : np.ndarray
            Output row indices for the warp destination.
        write_cols : np.ndarray
            Output column indices for the warp destination.
        read_i : np.ndarray
            Flattened indices into the raw sample grid for each warped pixel.
        """
        version_id = info.get("version_id")

        if version_id == 0:
            raise NotImplementedError

        elif version_id == 1:
            raise NotImplementedError

        elif version_id == 2:
            raise NotImplementedError

        elif version_id == 3:
            info["halffov"] = 14.4
            info["BeamCount"] = info["numbeams"]

            total_fov = 2 * info["halffov"]
            beam_width = total_fov / info["numbeams"]

            centers = np.linspace(
                -info["halffov"] + beam_width / 2,
                info["halffov"] - beam_width / 2,
                info["numbeams"],
            )

            left = centers - beam_width / 2
            right = centers + beam_width / 2

            beam_width_data = pd.DataFrame(
                {
                    "beam_num": np.arange(info["numbeams"]),
                    "beam_center": centers,
                    "beam_left": left,
                    "beam_right": right,
                }
            )

            # The following protocol is from
            # https://support.echoview.com/WebHelp/Reference/File_Formats/DIDSON_data_files.htm for DDF_03
            soundspeed = 1500  # Defaulted to the DIDSON specified sound speed of 1500/s

            is_high_res = info["resolution"] == 1
            is_serial_num_gt_18 = info["serialnumber"] > 18
            if is_high_res:
                delay_period = 0.000572 if is_serial_num_gt_18 else 0.000512
            else:
                delay_period = 0.001144 if is_serial_num_gt_18 else 0.001024

            info["windowstart"] = info["windowstart"] * delay_period * soundspeed / 2.0
            info["windowlength"] = (
                info["samplesperchannel"] * soundspeed / (2.0 * info["samplerate"])
            )

            sampleperiod = (1.0 / info["samplerate"]) * 1e6
            info.update(
                {
                    "beam_width_dir": os.path.abspath(beam_width_dir),
                    "beam_width_data": beam_width_data,
                    "sampleperiod": sampleperiod,
                    "soundspeed": soundspeed,
                    "samplesperbeam": info["samplesperchannel"],
                }
            )

        elif version_id == 4:
            # Convert windowlength code to meters
            info["windowlength"] = [1.25, 2.5, 5, 10, 20, 40][
                info["windowlength"] + 2 * (1 - info["resolution"])
            ]

            # Windowstart 1 to 31 times 0.75 (Lo) or 0.375 (Hi) or 0.419 for extended
            info["windowstart"] = 0.419 * info["windowstart"] * (2 - info["resolution"])

            info["halffov"] = 14.4

        elif version_id == 5:  # ARIS
            if info["pingmode"] in [1, 2]:
                BeamCount = 48
            elif info["pingmode"] in [3, 4, 5]:
                BeamCount = 96
            elif info["pingmode"] in [6, 7, 8]:
                BeamCount = 64
            elif info["pingmode"] in [9, 10, 11, 12]:
                BeamCount = 128
            else:
                raise

            WinStart = info["samplestartdelay"] * 0.000001 * info["soundspeed"] / 2

            info.update(
                {
                    "BeamCount": BeamCount,
                    "WinStart": WinStart,
                }
            )

            aris_frame = SimpleNamespace(**info)

            beam_width_data, camera_type = pyARIS.load_beam_width_data(
                frame=aris_frame, beam_width_dir=beam_width_dir
            )

            # What is the meter resolution of the smallest sample?
            min_pixel_size = pyARIS.get_minimum_pixel_meter_size(
                aris_frame, beam_width_data
            )

            # What is the meter resolution of the sample length?
            sample_length = (
                aris_frame.sampleperiod * 0.000001 * aris_frame.soundspeed / 2
            )

            # Choose the size of a pixel (or hard code it to some specific value)
            pixel_meter_size = max(min_pixel_size, sample_length)

            # Determine the image dimensions
            xdim, ydim, x_meter_start, y_meter_start, x_meter_stop, y_meter_stop = (
                pyARIS.compute_image_bounds(
                    pixel_meter_size,
                    aris_frame,
                    beam_width_data,
                    additional_pixel_padding_x=0,
                    additional_pixel_padding_y=0,
                )
            )

            if img_load_size is not None and ydim != img_load_size:
                print(
                    f"Default size {ydim}x{xdim} does not match img_load_size in the y-axis {img_load_size}"
                )
                scale_factor_y = img_load_size / ydim
                pixel_meter_size = pixel_meter_size / scale_factor_y
                xdim, ydim, x_meter_start, y_meter_start, x_meter_stop, y_meter_stop = (
                    pyARIS.compute_image_bounds(
                        pixel_meter_size,
                        aris_frame,
                        beam_width_data,
                        additional_pixel_padding_x=0,
                        additional_pixel_padding_y=0,
                    )
                )
                print(f"Reset the load size to {ydim}x{xdim}")

            if ixsize != -1:
                pixel_meter_size = pixel_meter_size * xdim / ixsize
                pixel_meter_size += 1e-5
                (
                    xdim,
                    ydim,
                    x_meter_start,
                    y_meter_start,
                    x_meter_stop,
                    y_meter_stop,
                ) = pyARIS.compute_image_bounds(
                    pixel_meter_size,
                    aris_frame,
                    beam_width_data,
                    additional_pixel_padding_x=0,
                    additional_pixel_padding_y=0,
                )

            read_rows, read_cols, write_rows, write_cols = (
                pyARIS.compute_mapping_from_sample_to_image(
                    pixel_meter_size,
                    xdim,
                    ydim,
                    x_meter_start,
                    y_meter_start,
                    aris_frame,
                    beam_width_data,
                )
            )

            read_i = read_rows * info["numbeams"] + info["numbeams"] - read_cols - 1

            pixel_meter_width = pixel_meter_size
            pixel_meter_height = pixel_meter_size

            info.update(
                {
                    "camera_type": camera_type,
                    "min_pixel_size": min_pixel_size,
                    "sample_length": sample_length,
                    "x_meter_start": x_meter_start,
                    "y_meter_start": y_meter_start,
                    "x_meter_stop": x_meter_stop,
                    "y_meter_stop": y_meter_stop,
                    "beam_width_dir": os.path.abspath(beam_width_dir),
                    "beam_width_data": beam_width_data,
                }
            )
        else:
            raise

        if version_id < 5:
            info["xdim"] = 300 if ixsize == -1 else ixsize
            ydim, xdim, write_rows, write_cols, read_i = DIDSON.mapscan(info)

            rmin = info["windowstart"]
            rmax = rmin + info["windowlength"]
            halffov_rad = np.radians(info["halffov"])

            pixel_meter_size = (2 * rmax * np.sin(halffov_rad)) / xdim
            pixel_meter_width = pixel_meter_size
            pixel_meter_height = pixel_meter_size

            x_meter_start = -rmax * np.sin(halffov_rad)
            x_meter_stop = rmax * np.sin(halffov_rad)
            y_meter_start = rmax
            y_meter_stop = rmin * np.cos(halffov_rad)

        unwarped_shape = [
            info["samplesperchannel"],
            info["numbeams"],
        ]

        write_rows = write_rows
        write_cols = write_cols
        read_i = read_i

        info.update(
            {
                "xdim": xdim,
                "ydim": ydim,
                "pixel_meter_width": pixel_meter_width,
                "pixel_meter_height": pixel_meter_height,
                "pixel_meter_size": pixel_meter_size,
                "x_meter_start": x_meter_start,
                "x_meter_stop": x_meter_stop,
                "y_meter_start": y_meter_start,
                "y_meter_stop": y_meter_stop,
                "unwarped_shape": unwarped_shape,
            }
        )

        # Fix common but critical corruption errors
        if info["startframe"] > 65535:
            info["startframe"] = 0
        if info["endframe"] > 65535:
            info["endframe"] = 0

        # Record the proportion of measurements that are present in the warp (increases as xdim increases)
        info["proportion_warp"] = len(np.unique(read_i)) / (
            info["numbeams"] * info["samplesperchannel"]
        )

        if info["proportion_warp"] > 0.01:
            warnings.warn(
                f'{info["proportion_warp"]*100:.2f}% of sensor readings are not being used'
            )
        if unwarped_shape[0] < ydim:
            warnings.warn(
                f"The warped image is shorter than the unwarped image {ydim} compared to {unwarped_shape[0]}"
            )

        return info, write_rows, write_cols, read_i

    @staticmethod
    def lens_distortion(nbeams: int, theta: np.ndarray):
        """Map beam angles to beam indices using the empirical lens model.

        Parameters
        ----------
        nbeams : int
            Number of sonar beams.
        theta : (A,) ndarray
            Angle of warp for each x index.

        Returns
        -------
        beamnum : (A,) ndarray
            Distortion-adjusted beam index for each input angle.

        """

        factor, a = {
            48: [1, [0.0015, -0.0036, 1.3351, 24.0976]],
            189: [4.026, [0.0015, -0.0036, 1.3351, 24.0976]],
            96: [1.012, [0.0030, -0.0055, 2.6829, 48.04]],
            381: [4.05, [0.0030, -0.0055, 2.6829, 48.04]],
        }[nbeams]
        beamnum = np.rint(
            factor * (a[0] * theta**3 + a[1] * theta**2 + a[2] * theta + a[3]) + 1
        )
        beamnum = np.clip(
            beamnum, 0, np.iinfo(np.uint32).max
        )  # MAH 2025-02-14 12:16:51 issue #31: this is required to silence a warning for the negative values in
        # beam_num being cast to 0. This line mimics the previous behaviour (clipping the negative values) because
        # they are floats. If they were ints this would take the 2s compliment
        beamnum = beamnum.astype(np.uint32)

        return beamnum

    @staticmethod
    def mapscan(info: dict):
        """Build the DIDSON sample-to-image warp lookup for one frame geometry.

        Parameters
        ----------
        info : dict
            Parsed and derived DIDSON metadata containing range, beam, and output
            geometry fields.

        Returns
        -------
        ydim : int
            y-dimension of warped image.
        xdim : int
            x-dimension of warped image.
        write_rows : (A,) ndarray, np.uint16
            Row indices to write to warped image.
        write_cols : (A,) ndarray, np.uint16
            Column indices to write to warped image.
        read_i : (A,) ndarray, np.uint32
            Indices to read from raw sonar measurements.

        """

        xdim = info.get("xdim", 0)
        rmin = info.get("windowstart", 0)
        rmax = rmin + info.get("windowlength", 0)
        halffov = info.get("halffov", 0)
        nbeams = info.get("numbeams", 0)
        nbins = info.get("samplesperchannel", 0)

        degtorad = 3.14159 / 180  # conversion of degrees to radians
        radtodeg = 180 / 3.14159  # conversion of radians to degrees

        d2 = rmax * np.cos(
            halffov * degtorad
        )  # see drawing (distance from point scan touches image boundary to origin)
        d3 = rmin * np.cos(
            halffov * degtorad
        )  # see drawing (bottom of image frame to r,theta origin in meters)
        c1 = (nbins - 1) / (
            rmax - rmin
        )  # precalcualtion of constants used in do loop below
        c2 = (nbeams - 1) / (2 * halffov)

        gamma = xdim / (
            2 * rmax * np.sin(halffov * degtorad)
        )  # Ratio of pixel number to position in meters
        ydim = int(
            np.fix(gamma * (rmax - d3) + 0.5)
        )  # number of pixels in image in vertical direction
        svector = np.zeros(
            xdim * ydim, dtype=np.uint32
        )  # make vector and fill in later
        ix = np.arange(1, xdim + 1)  # pixels in x dimension
        x = ((ix - 1) - xdim / 2) / gamma  # convert from pixels to meters

        for iy in range(1, ydim + 1):
            y = rmax - (iy - 1) / gamma  # convert from pixels to meters
            r = np.sqrt(y**2 + x**2)  # convert to polar cooridinates
            theta = radtodeg * np.arctan2(x, y)  # theta is in degrees
            binnum = np.rint((r - rmin) * c1 + 1.5)  # the rangebin number
            binnum = np.clip(
                binnum, 0, np.iinfo(np.uint32).max
            )  # MAH 2025-02-14 12:16:51 issue #31: this is required to silence a warning for the negative values in
            # beam_num being cast to 0. This line mimics the previous behaviour (clipping the negative values)
            # because they are floats. If they were ints this would take the 2s compliment
            binnum = binnum.astype(np.uint32)  # the rangebin number
            beamnum = DIDSON.lens_distortion(
                nbeams, theta
            )  # remove lens distortion using empirical formula

            # find position in sample array expressed as a vector
            # make pos = 0 if outside sector, else give it the offset in the sample array
            pos = (
                (beamnum > 0)
                * (beamnum <= nbeams)
                * (binnum > 0)
                * (binnum <= nbins)
                * ((beamnum - 1) * nbins + binnum)
            )
            svector[(ix - 1) * ydim + iy - 1] = (
                pos  # The offset in this array is the pixel offset in the image array
            )
            # The value at this offset is the offset in the sample array

        svector = svector.reshape(xdim, ydim).T.flat
        svectori = svector != 0

        read_i = np.flipud(
            np.arange(nbins * nbeams, dtype=np.uint32).reshape(nbins, nbeams).T
        ).flat[svector[svectori] - 1]
        write_rows, write_cols = np.unravel_index(np.where(svectori)[0], (ydim, xdim))
        return (
            ydim,
            xdim,
            write_rows.astype(np.uint16),
            write_cols.astype(np.uint16),
            read_i,
        )

    def __FasterDIDSONRead(self, file, start_frame, end_frame):
        """Read a contiguous block of raw sonar frames without applying the warp.

        Parameters
        ----------
        file : file-like object, string, or pathlib.Path
            DIDSON or ARIS file to read.
        start_frame : int
            Zero-indexed start of frame range (inclusive).
        end_frame : int
            End of frame range (exclusive).

        Returns
        -------
        raw_frames : (end_frame - start_frame, framesize) ndarray, np.uint8
            Extracted and flattened raw sonar measurements for frame range.

        """

        if hasattr(file, "read"):
            file_ctx = contextlib.nullcontext(file)
        else:
            file_ctx = open(file, "rb")

        with file_ctx as fid:
            framesize = self.info["framesize"]
            frameheadersize = self.info["frameheadersize"]

            fid.seek(
                self.info["fileheadersize"]
                + start_frame * (frameheadersize + framesize)
                + frameheadersize,
                0,
            )

            # Read data from the first frame
            first_frame_data = np.frombuffer(
                fid.read(framesize + frameheadersize)[:framesize], dtype=np.uint8
            )

            # Possible byte misalignment if the data from the first frames is empty/zero. This can mean we are working
            # with a modified, shortened clip.
            if np.all(first_frame_data == 0):
                warnings.warn(
                    f"First frame at start_frame={start_frame} contains only zeroes. "
                    "This may indicate a shorted clip (modified version of the original DIDSON/ARIS file."
                    f"Resetting start_frame to 0 and adjusting the end_frame to {end_frame}."
                )
                end_frame = end_frame - start_frame + 1  # inclusive
                start_frame = 0

            fid.seek(
                self.info["fileheadersize"]
                + start_frame * (frameheadersize + framesize)
                + frameheadersize,
                0,
            )

            frames = []
            frame_count = 0
            while end_frame == 0 or frame_count < (end_frame - start_frame):
                frame_data = fid.read(framesize + frameheadersize)

                if not frame_data:
                    warnings.warn(
                        f"Warning: No more frame data to read at index {frame_count}. Exiting loop."
                    )
                    break

                frame = np.frombuffer(frame_data[:framesize], dtype=np.uint8)
                if frame.shape[0] != framesize:
                    warnings.warn(
                        f"Warning: Invalid frame size detected after unpacking frame data (expected {framesize}, got"
                        f" {frame.shape[0]})."
                        f" Exiting loop."
                    )
                    break

                frames.append(np.frombuffer(frame_data[:framesize], dtype=np.uint8))
                frame_count += 1

            return np.array(frames, dtype=np.uint8)

    def precompute_area_like_pix_and_count(
        self, big_write_rows, big_write_cols, big_ydim, big_xdim, out_ydim, out_xdim
    ):
        """Precompute resized-pixel assignments for area-style RMS downsampling.

        Parameters
        ----------
        big_write_rows, big_write_cols : np.ndarray
            Pixel coordinates in the larger warped image.
        big_ydim, big_xdim : int
            Height and width of the larger warped image.
        out_ydim, out_xdim : int
            Height and width of the final output image.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Flat output-pixel indices for each source sample and per-pixel sample
            counts with a minimum value of ``1`` for safe division.
        """
        # Project big pixel centers into small grid coordinates
        y_f = (big_write_rows.astype(np.float32) + 0.5) * (out_ydim / big_ydim) - 0.5
        x_f = (big_write_cols.astype(np.float32) + 0.5) * (out_xdim / big_xdim) - 0.5

        small_r = np.floor(y_f).astype(np.int64)
        small_c = np.floor(x_f).astype(np.int64)

        small_r = np.clip(small_r, 0, out_ydim - 1)
        small_c = np.clip(small_c, 0, out_xdim - 1)

        pix = (small_r * out_xdim + small_c).astype(np.int64)
        n_pix = out_ydim * out_xdim

        count = np.bincount(pix, minlength=n_pix).astype(np.float32)
        count_safe = np.maximum(count, 1.0).astype(np.float32)

        # This should always be exactly n_pix if pix is in range
        if count_safe.shape[0] != n_pix:
            raise RuntimeError(f"count length {count_safe.shape[0]} != n_pix {n_pix}")

        return pix, count_safe

    def sanitize_mapping(self, read_i, pix, Nraw, extra_arrays=()):
        """Drop mapping entries that point outside the raw flattened frame buffer.

        Parameters
        ----------
        read_i : np.ndarray
            Flattened raw-sample indices referenced by the warp mapping.
        pix : np.ndarray
            Flattened output-pixel indices paired with ``read_i``.
        Nraw : int
            Number of raw samples available per frame.
        extra_arrays : tuple, optional
            Additional arrays aligned with ``read_i`` that should be filtered using
            the same validity mask.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[np.ndarray]]
            Filtered ``read_i``, filtered ``pix``, and any filtered companion arrays.
        """
        read_i = np.asarray(read_i)
        pix = np.asarray(pix)

        valid = (read_i >= 0) & (read_i < Nraw)
        if not np.all(valid):
            bad = np.count_nonzero(~valid)
            # Keep this print/log if you want to monitor mapping quality
            print(
                f"MAH sanitize_mapping: dropping {bad}/{read_i.size} out-of-bounds read_i"
            )

        read_i_f = read_i[valid].astype(np.int64)
        pix_f = pix[valid].astype(np.int64)

        filtered_extras = []
        for a in extra_arrays:
            a = np.asarray(a)
            filtered_extras.append(a[valid])

        return read_i_f, pix_f, filtered_extras

    def warp_samples_to_rms_image(
        self, data, read_i, pix, count_safe, out_ydim, out_xdim
    ):
        """
        Map raw sample values into an output image grid using RMS aggregation.

        For each frame in `data`, this function gathers raw sample values using
        `read_i` and assigns each gathered value to an output pixel specified by
        `pix`. When multiple raw samples map to the same output pixel, the pixel
        value is computed as the root mean square (RMS) of those samples:

            output_pixel = sqrt(mean(sample_values ** 2))

        This is useful when the output image should represent signal magnitude or
        energy rather than a signed average. Positive and negative values therefore
        contribute equally and do not cancel each other out.

        Invalid mapping entries are removed by `sanitize_mapping` before the warp is
        computed. Pixel counts are recomputed after filtering so that the RMS
        normalization reflects only valid mappings.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (T, Nraw), where T is the number of frames or time steps
            and Nraw is the number of raw samples per frame.
        read_i : np.ndarray
            Integer array of shape (A,) containing indices into the raw sample axis
            of `data`.
        pix : np.ndarray
            Integer array of shape (A,) containing flattened output pixel indices
            in the range [0, out_ydim * out_xdim).
        count_safe : np.ndarray
            Unused. Counts are recomputed internally after invalid mappings are
            removed. This argument is retained only for API compatibility.
        out_ydim : int
            Height of the output image.
        out_xdim : int
            Width of the output image.

        Returns
        -------
        np.ndarray
            Array of shape (T, out_ydim, out_xdim) with dtype uint8. Each output
            pixel contains the clipped RMS value of all mapped raw samples for that
            frame. Pixels with no mapped samples are set to 0.
        """

        Nraw = data.shape[1]
        n_pix = out_ydim * out_xdim

        # Filter mapping entries that would index out of bounds
        read_i_f, pix_f, _ = self.sanitize_mapping(read_i, pix, Nraw)

        # Recompute counts based on filtered pix (important!)
        count = np.bincount(pix_f, minlength=n_pix).astype(np.float32)
        count_safe_f = np.maximum(count, 1.0).astype(np.float32)
        inv_count = (1.0 / count_safe_f).astype(np.float32)

        out = np.empty((data.shape[0], out_ydim, out_xdim), dtype=np.uint8)

        for t in range(data.shape[0]):
            v = data[t, read_i_f].astype(np.float32)
            s2 = np.bincount(pix_f, weights=v * v, minlength=n_pix).astype(np.float32)
            img = np.sqrt(s2 * inv_count).reshape(out_ydim, out_xdim)
            out[t] = np.clip(img, 0, 255).astype(np.uint8)

        return out

    def load_raw_data(self, file=None, start_frame=-1, end_frame=-1):
        """Load raw flattened frame data using the file's default frame bounds.

        Parameters
        ----------
        file : file-like object, str, or pathlib.Path, optional
            File to read. Defaults to the file recorded in ``self.info``.
        start_frame : int, optional
            Inclusive start frame. ``-1`` uses the file header's start frame.
        end_frame : int, optional
            Exclusive end frame. ``-1`` uses the header end frame or total frame
            count when the header does not provide one.

        Returns
        -------
        np.ndarray
            Raw frame matrix with shape ``(num_frames, framesize)``.
        """
        if file is None:
            file = self.info["filename"]

        if hasattr(file, "read"):
            file_ctx = contextlib.nullcontext(file)
        else:
            file = Path(file).expanduser().resolve()
            file_ctx = open(file, "rb")

        with file_ctx as fid:
            fid.seek(0)  # Reset pointer to start
            svector = None
            if start_frame == -1:
                start_frame = self.info["startframe"]
            if end_frame == -1:
                end_frame = self.info["endframe"] or self.info["numframes"]

            data = self.__FasterDIDSONRead(fid, start_frame, end_frame)
            return data

    def load_frames(
        self, file=None, start_frame=0, end_frame=-1, return_unwarped=False
    ):
        """Load raw frames, optionally expose the unwarped view, and return warped images.

        Parameters
        ----------
        file : file-like object, string, or pathlib.Path, optional
            The DIDSON or ARIS file to read. Defaults to `filename` in `info`.
        start_frame : int, optional
            Zero-indexed start of frame range (inclusive). Defaults to the first available.
        end_frame : int, optional
            End of frame range (exclusive). Defaults to the last available frame.
        return_unwarped : bool, optional
            If ``True``, also return the raw frame data reshaped into
            ``(samplesperchannel, numbeams)`` order.

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None, np.ndarray | None]
            Warped frames, optional unwarped frames, and optional full-resolution
            warped frames when ``return_original_image`` was enabled at init time.

        """
        data = self.load_raw_data(file, start_frame, end_frame)
        if return_unwarped:
            unwarped_frames_shape = [
                data.shape[0],
                self.info["unwarped_shape"][0],
                self.info["unwarped_shape"][1],
            ]
            unwarped_frames = np.reshape(
                data,
                unwarped_frames_shape,
            )
            unwarped_frames = unwarped_frames[
                :, ::-1, ::-1
            ].copy()  # MAH 2025-02-05 19:11:09 I have no idea why this copy is needed but you get a negative
            # indexing error without it
        else:
            unwarped_frames = None

        frames = self.warp_samples_to_rms_image(
            data=data,
            read_i=self.read_i,
            pix=self._pix_area,
            count_safe=self._count_area,  # not used after sanitize+recount, but ok to pass if you keep signature
            out_ydim=self.out_ydim,
            out_xdim=self.out_xdim,
        )

        if self.return_original_image:
            original_frames = np.zeros(
                (data.shape[0], self.big_ydim, self.big_xdim), dtype=np.uint8
            )
            original_frames[:, self.big_write_rows, self.big_write_cols] = data[
                :, self.big_read_i
            ]
        else:
            original_frames = None

        return frames, unwarped_frames, original_frames

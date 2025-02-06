"""
Utilities to read and produce to-scale images from DIDSON and ARIS sonar files.

Portions of this code were adapted from SoundMetrics MATLAB code.

@author kulits
"""

__version__ = "b1.0.2"

import contextlib
import numpy as np
import os
import struct
from types import SimpleNamespace
from pathlib import Path

from . import pyARIS
from .pyDIDSON_format import *

BASE = Path(__file__).parent.parent.parent
BEAM_WIDTH_DIR = (BASE / "beam_widths").resolve()


class DIDSON:
    def __init__(self, file, beam_width_dir=BEAM_WIDTH_DIR, ixsize=-1):
        """Load header info from DIDSON file and precompute some warps.

        Parameters
        ----------
        file : file-like object, string, or pathlib.Path
            The DIDSON or ARIS file to read.
        beam_width_dir : string or pathlib.Path, optional
            Location of ARIS beam width CSV files. Only used for ARIS files.
        ixsize : int, optional
            x-dimension width of output warped images to produce. Width is approximate for ARIS files and definite for
            DIDSON. If not specified, the default for ARIS is determined by pyARIS and the default for DIDSON is 300.

        Returns
        -------
        info : dict
            Dictionary of extracted headers and computed sonar values.

        """

        if hasattr(file, "read"):
            file_ctx = contextlib.nullcontext(file)
        else:
            file_ctx = open(file, "rb")

        with file_ctx as fid:
            assert fid.read(3) == b"DDF"

            version_id = fid.read(1)[0]

            fid.seek(0)

            info = {
                "pydidson_version": __version__,
            }
            self.info = info

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

            if version_id == 0:
                raise NotImplementedError
            elif version_id == 1:
                raise NotImplementedError
            elif version_id == 2:
                raise NotImplementedError
            elif version_id == 3:
                # Convert windowlength code to meters
                info["windowlength"] = {
                    0b00: [0.83, 2.5, 5, 10, 20, 40],  # DIDSON-S, Extended Windows
                    0b01: [1.125, 2.25, 4.5, 9, 18, 36],  # DIDSON-S, Classic Windows
                    0b10: [2.5, 5, 10, 20, 40, 70],  # DIDSON-LR, Extended Window
                    0b11: [2.25, 4.5, 9, 18, 36, 72],  # DIDSON-LR, Classic Windows
                }[info["configflags"] & 0b11][
                    info["windowlength"] + 2 * (1 - info["resolution"])
                ]

                # Windowstart 1 to 31 times 0.75 (Lo) or 0.375 (Hi) or 0.419 for extended
                info["windowstart"] = {
                    0b0: 0.419
                    * info["windowstart"]
                    * (2 - info["resolution"]),  # meters for extended DIDSON
                    0b1: 0.375 * info["windowstart"] * (2 - info["resolution"]),
                    # meters for standard or long range DIDSON
                }[info["configflags"] & 0b1]

                info["halffov"] = 14.4
            elif version_id == 4:
                # Convert windowlength code to meters
                info["windowlength"] = [1.25, 2.5, 5, 10, 20, 40][
                    info["windowlength"] + 2 * (1 - info["resolution"])
                ]

                # Windowstart 1 to 31 times 0.75 (Lo) or 0.375 (Hi) or 0.419 for extended
                info["windowstart"] = (
                    0.419 * info["windowstart"] * (2 - info["resolution"])
                )

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
                    }
                )
            else:
                raise

            if version_id < 5:
                info["xdim"] = 300 if ixsize == -1 else ixsize
                ydim, xdim, write_rows, write_cols, read_i = self.__mapscan()

                # widthscale meters/pixels
                pixel_meter_width = (
                    2
                    * (info["windowstart"] + info["windowlength"])
                    * np.sin(np.radians(14.25))
                    / xdim
                )
                # heightscale meters/pixels
                pixel_meter_height = (
                    (info["windowstart"] + info["windowlength"])
                    - info["windowstart"] * np.cos(np.radians(14.25))
                ) / ydim

                pixel_meter_size = (pixel_meter_width + pixel_meter_height) / 2

            self.write_rows = write_rows
            self.write_cols = write_cols
            self.read_i = read_i

            info.update(
                {
                    "xdim": xdim,
                    "ydim": ydim,
                    "pixel_meter_width": pixel_meter_width,
                    "pixel_meter_height": pixel_meter_height,
                    "pixel_meter_size": pixel_meter_size,
                }
            )

            # Fix common but critical corruption errors
            if info["startframe"] > 65535:
                info["startframe"] = 0
            if info["endframe"] > 65535:
                info["endframe"] = 0

            try:
                info["filename"] = os.path.abspath(file_ctx.name)
            except AttributeError:
                info["filename"] = None

            # Record the proportion of measurements that are present in the warp (increases as xdim increases)
            info["proportion_warp"] = len(np.unique(read_i)) / (
                info["numbeams"] * info["samplesperchannel"]
            )

    def __lens_distortion(self, nbeams, theta):
        """Removes Lens distortion determined by empirical work at the barge.

        Parameters
        ----------
        nbeams : int
            Number of sonar beams.
        theta : (A,) ndarray
            Angle of warp for each x index.

        Returns
        -------
        beamnum : (A,) ndarray
            Distortion-adjusted beam number for each theta.

        """

        factor, a = {
            48: [1, [0.0015, -0.0036, 1.3351, 24.0976]],
            189: [4.026, [0.0015, -0.0036, 1.3351, 24.0976]],
            96: [1.012, [0.0030, -0.0055, 2.6829, 48.04]],
            381: [4.05, [0.0030, -0.0055, 2.6829, 48.04]],
        }[nbeams]

        return np.rint(
            factor * (a[0] * theta**3 + a[1] * theta**2 + a[2] * theta + a[3]) + 1
        ).astype(np.uint32)

    def __mapscan(self):
        """Calculate warp mapping from raw to scale images.

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

        xdim = self.info["xdim"]
        rmin = self.info["windowstart"]
        rmax = rmin + self.info["windowlength"]
        halffov = self.info["halffov"]
        nbeams = self.info["numbeams"]
        nbins = self.info["samplesperchannel"]

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
            binnum = np.rint((r - rmin) * c1 + 1.5).astype(
                np.uint32
            )  # the rangebin number
            beamnum = self.__lens_distortion(
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
        """Load raw frames from DIDSON.

        Parameters
        ----------
        file : file-like object, string, or pathlib.Path
            The DIDSON or ARIS file to read.
        info : dict
            Dictionary of extracted headers and computed sonar values.
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
            framesize = self.info["samplesperchannel"] * self.info["numbeams"]
            frameheadersize = self.info["frameheadersize"]

            fid.seek(
                self.info["fileheadersize"]
                + start_frame * (frameheadersize + framesize)
                + frameheadersize,
                0,
            )

            return np.array(
                [
                    np.frombuffer(
                        fid.read(framesize + frameheadersize)[:framesize],
                        dtype=np.uint8,
                    )
                    for _ in range(end_frame - start_frame)
                ],
                dtype=np.uint8,
            )

    def load_raw_data(self, file=None, start_frame=-1, end_frame=-1):
        """
        Public interface to self.__FasterDIDSONRead
        """
        if file is None:
            file = self.info["filename"]

        if hasattr(file, "read"):
            file_ctx = contextlib.nullcontext(file)
        else:
            file_ctx = open(file, "rb")

        with file_ctx as fid:
            svector = None
            if start_frame == -1:
                start_frame = self.info["startframe"]
            if end_frame == -1:
                end_frame = self.info["endframe"] or self.info["numframes"]

            data = self.__FasterDIDSONRead(fid, start_frame, end_frame)
            return data

    def load_frames(self, file=None, start_frame=-1, end_frame=-1):
        """Load and warp DIDSON frames into images.

        Parameters
        ----------
        file : file-like object, string, or pathlib.Path, optional
            The DIDSON or ARIS file to read. Defaults to `filename` in `info`.
        start_frame : int, optional
            Zero-indexed start of frame range (inclusive). Defaults to the first available.
        end_frame : int, optional
            End of frame range (exclusive). Defaults to the last available frame.

        Returns
        -------
        frames : (end_frame - start_frame, ydim, xdim) ndarray, np.uint8
            Warped-to-scale sonar image tensor.

        """
        data = self.load_raw_data(file, start_frame, end_frame)
        frames = np.zeros(
            (data.shape[0], self.info["ydim"], self.info["xdim"]), dtype=np.uint8
        )
        frames[:, self.write_rows, self.write_cols] = data[:, self.read_i]

        return frames

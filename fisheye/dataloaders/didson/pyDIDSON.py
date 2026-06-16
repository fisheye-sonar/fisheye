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
import structlog

from . import pyARIS
from .pyDIDSON_format import *
from fisheye.dataloaders.echogram import (
    compute_echogram,
    echogram_uses_bg_subtraction,
)
from fisheye.dataloaders.compute_bg_subtraction import compute_bg_subtraction

BASE = Path(__file__).parent.parent.parent
BEAM_WIDTH_DIR = (BASE / "beam_widths").resolve()
logger = structlog.get_logger(__name__)


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

        info = self.read_header(file)
        self.info, self.write_rows, self.write_cols, self.read_i = (
            DIDSON.compute_image_metadata(info, beam_width_dir, ixsize)
        )

    def read_header(self, file: Union[str, Path]):
        """Reads the header from a DIDSON or ARIS file.

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
        info: dict, beam_width_dir: Path = BEAM_WIDTH_DIR, ixsize: int = -1
    ):
        """Computes derived sonar parameters, and precomputes pixel mapping for warping sonar data into to-scale images.

        Parameters
        ----------
        info : dict
            Parsed header information.
        beam_width_dir : Path
            Path to beam width CSV files (for ARIS).
        ixsize : int
            Target x-dimension (overrides default if provided).

        Returns
        -------
        info : dict
            Updated info with computed image metadata.
        write_rows : np.ndarray
            Row indices for writing warped images.
        write_cols : np.ndarray
            Column indices for writing warped images.
        read_i : np.ndarray
            Indices into unwarped image array.
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

    def load_raw_data(self, file=None, start_frame=-1, end_frame=-1):
        """
        Public interface to self.__FasterDIDSONRead
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
        self,
        file=None,
        start_frame=0,
        end_frame=-1,
        return_unwarped=False,
        return_warped=True,
    ):
        """Load and warp DIDSON frames into images.

        Parameters
        ----------
        file : file-like object, string, or pathlib.Path, optional
            The DIDSON or ARIS file to read. Defaults to `filename` in `info`.
        start_frame : int, optional
            Zero-indexed start of frame range (inclusive). Defaults to the first available.
        end_frame : int, optional
            End of frame range (exclusive). Defaults to the last available frame.
        return_unwarped : bool, optional
            Whether to return the unwarped frames.
        return_warped : bool, optional
            Whether to return the warped frames.
        Returns
        -------
        frames : (end_frame - start_frame, ydim, xdim) ndarray, np.uint8 or None
            Warped-to-scale sonar image tensor, or None when ``return_warped`` is False.

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
        if not return_warped:
            frames = None
        else:
            frames = np.zeros(
                (data.shape[0], self.info["ydim"], self.info["xdim"]), dtype=np.uint8
            )
            frames[:, self.write_rows, self.write_cols] = data[:, self.read_i]

        return frames, unwarped_frames

    def load_echogram(
        self,
        file=None,
        start_frame=0,
        end_frame=-1,
        num_frames_bg_subtract=1000,
        use_blur=True,
        use_multithreading=True,
        max_workers=2,
        echogram_channels=None,
    ):
        """Load unwarped beam data and return an echogram without building warped frames.
        Parameters
        ----------
        file : file-like object, string, or pathlib.Path, optional
            The DIDSON or ARIS file to read. Defaults to ``filename`` in ``info``.
        start_frame, end_frame : int, optional
            Zero-indexed frame range ``[start_frame, end_frame)``. ``end_frame=-1`` uses
            the last frame in the file.
        num_frames_bg_subtract : int, optional
            Maximum number of leading frames used to estimate background statistics.
        use_blur, use_multithreading, max_workers : optional
            Background estimation options (same as :class:`BaseDataset`).
        echogram_channels : list, optional
            Ordered channel selection, using entries from
            ``{"bgs", "bgs_angle", "angle", "raw", "center_line", "0", None}``.
        Returns
        -------
        echogram : ndarray, shape (num_frames, height, num_channels), float32
            Echogram for the requested frame range. The last loaded frame is dropped to
            match :class:`BaseDataset` behaviour.
        """
        logger.info(
            "loading_echogram",
            file=str(file) if file is not None else self.info.get("filename"),
            start_frame=start_frame,
            end_frame=end_frame,
        )
        if start_frame == -1:
            start_frame = self.info.get("startframe", 0)
        if end_frame == -1:
            end_frame = max(
                x for x in (self.info.get("endframe"), self.info.get("numframes")) if x
            )
        logger.info(
            "updated_echogram_frame_range",
            start_frame=start_frame,
            end_frame=end_frame,
        )

        do_bg_subtract_echogram = echogram_uses_bg_subtraction(echogram_channels)
        mean_blurred_frame = None
        mean_normalization_value = None
        if do_bg_subtract_echogram:
            num_frames_bg = min(end_frame - start_frame, num_frames_bg_subtract)
            _, unwarped_bg = self.load_frames(
                file,
                start_frame,
                start_frame + num_frames_bg,
                return_unwarped=True,
                return_warped=False,
            )
            mean_blurred_frame, mean_normalization_value = compute_bg_subtraction(
                unwarped_bg,
                use_blur=use_blur,
                use_multithreading=use_multithreading,
                max_workers=max_workers,
            )

        _, unwarped_frames = self.load_frames(
            file,
            start_frame,
            end_frame,
            return_unwarped=True,
            return_warped=False,
        )
        unwarped_frames = unwarped_frames[:-1]

        return compute_echogram(
            unwarped_frames,
            mean_blurred_frame=mean_blurred_frame,
            mean_normalization_value=mean_normalization_value,
            echogram_channels=echogram_channels,
        )

import hashlib
import json
import warnings

import cv2
import numpy as np
from torch.utils.data import Dataset

from fisheye.common.generic import run_with_threads
from fisheye.configs import BaseDatasetConfig
from fisheye.endpoints.config import validate_capture_config


class BaseDataset(Dataset):
    """BaseDataset

    Base class for datasets used in model inference. This class is initialized with a BaseDatasetConfig object
    containing all necessary configuration parameters such as frame ranges, background subtraction settings,
    and batch size.
    """

    def __init__(self, config: BaseDatasetConfig):
        """
        Initializes the dataset using the provided configuration.

        Args:
            config (BaseDatasetConfig): Configuration object containing dataset parameters.
        """

        self.start_frame = config.start_frame
        self.end_frame = config.end_frame
        self.batch_size = config.batch_size
        self.cache_bg_frames = config.cache_bg_frames
        self.num_frames_bg_subtract = config.num_frames_bg_subtract
        self.do_bg_subtract = config.do_bg_subtract
        self.only_pos_bgs = config.only_pos_bgs
        self.return_echogram_with_bg_subtracted = (
            config.return_echogram_with_bg_subtracted
        )
        self.return_unwarped = config.return_unwarped
        self.return_echogram = config.return_echogram
        self.max_workers = config.max_workers
        self.use_multithreading = config.use_multithreading
        self.use_blur = config.use_blur
        self.return_original_image = config.return_original_image
        self.capture_endpoint_frames = config.capture_endpoint_frames
        if self.capture_endpoint_frames:
            validate_capture_config(config)
        if self.capture_endpoint_frames and self.cache_bg_frames:
            warnings.warn(
                "cache_bg_frames is disabled during endpoint capture; "
                "the bounded endpoint frame store owns retained images",
                UserWarning,
                stacklevel=2,
            )
            self.cache_bg_frames = False
        # Key by requested dataset index, rather than append order: workers and
        # repeated/noncontiguous requests do not necessarily visit frames in order.
        self._frame_cache = {}

        self._init_bg_frame()

    def _init_bg_frame(self):
        """Initialize background frame for subtraction."""
        if self.do_bg_subtract or self.return_echogram:
            num_frames_bg = min(
                self.end_frame - self.start_frame,
                self.num_frames_bg_subtract // self.batch_size * self.batch_size + 1,
            )
            frames_for_bg_subtract, unwarped_frames_for_bg_subtract = self.load_frames(
                self.start_frame,
                self.start_frame + num_frames_bg,
                return_unwarped=self.return_unwarped or self.return_echogram,
            )
            if self.return_unwarped or self.return_echogram:
                (
                    self.unwarped_mean_blurred_frame,
                    self.unwarped_mean_normalization_value,
                ) = self._compute_bg_subtraction(unwarped_frames_for_bg_subtract)

        if self.do_bg_subtract:

            self.mean_blurred_frame, self.mean_normalization_value = (
                self._compute_bg_subtraction(frames_for_bg_subtract)
            )
            if self.capture_endpoint_frames:
                self._describe_endpoint_preprocessing(frames_for_bg_subtract)

    def _describe_endpoint_preprocessing(self, background_frames):
        """Record the detector's existing BGS statistics without recomputing them."""
        background_ids = self._loaded_frame_ids(
            self.start_frame, len(background_frames)
        )
        normalization = self.mean_normalization_value
        if not np.isfinite(normalization) or normalization < 0:
            raise ValueError(
                "Endpoint BGS normalization must be finite and nonnegative"
            )
        preprocessing = {
            "profile": "fisheye_signed_bgs_v1",
            "representation": "bgs_signed_u8",
            "coordinate_space": "warped_original",
            "dtype": "uint8",
            "layout": "HW",
            "shape": list(background_frames.shape[1:]),
            "use_blur": False,
            "only_pos_bgs": False,
            "scale": 255.0,
            "background_frame_ids": list(background_ids),
            "background_frame_count": len(background_ids),
            "background_num_frames_requested": self.num_frames_bg_subtract,
            "background_batch_size": self.batch_size,
            "normalization": float(normalization) if normalization > 0 else 255.0,
            "sampled_normalization": float(normalization),
            "normalization_policy": (
                "sample_positive_residual_range"
                if normalization > 0
                else "uint8_range_fallback"
            ),
            "zero_normalization": bool(normalization == 0),
            "background_mean_sha256": hashlib.sha256(
                np.ascontiguousarray(self.mean_blurred_frame).tobytes()
            ).hexdigest(),
            "source": "detector_center_channel",
            "source_channel": 1,
            "signed_encoding": "clip(((frame-background)/normalization+1)/2*255,0,255)",
            "background_shared_with_detector": True,
        }
        preprocessing["fingerprint"] = hashlib.sha256(
            json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.endpoint_preprocessing = preprocessing

    def _compute_bg_subtraction(self, frames_for_bg_subtract):
        """Calculate the mean blurred frame and normalization value."""
        if len(frames_for_bg_subtract) == 0:
            raise ValueError("Cannot compute background statistics from no frames")
        if not self.use_blur:
            mean_blurred_frame = np.mean(frames_for_bg_subtract, axis=0)
            # Decoded sonar is uint8: abs would copy the full background stack.
            maximum_source = (
                frames_for_bg_subtract
                if np.issubdtype(frames_for_bg_subtract.dtype, np.unsignedinteger)
                else np.abs(frames_for_bg_subtract)
            )
            max_blurred_frame = np.max(maximum_source, axis=0).astype(np.float64)

        else:
            mean_blurred_frame = np.zeros(
                [frames_for_bg_subtract.shape[1], frames_for_bg_subtract.shape[2]],
                dtype=np.float32,
            )
            max_blurred_frame = np.zeros(
                [frames_for_bg_subtract.shape[1], frames_for_bg_subtract.shape[2]],
                dtype=np.float32,
            )
            if self.use_multithreading:
                blurred_frames = run_with_threads(
                    lambda i: cv2.GaussianBlur(frames_for_bg_subtract[i], (5, 5), 0),
                    list(range(frames_for_bg_subtract.shape[0])),
                    max_workers=self.max_workers,
                )
                # Aggregate results
                for blurred in blurred_frames:
                    mean_blurred_frame += blurred
                    max_blurred_frame = np.maximum(max_blurred_frame, np.abs(blurred))
            else:
                for i in range(frames_for_bg_subtract.shape[0]):
                    blurred = cv2.GaussianBlur(frames_for_bg_subtract[i], (5, 5), 0)
                    mean_blurred_frame += blurred
                    max_blurred_frame = np.maximum(max_blurred_frame, np.abs(blurred))
            mean_blurred_frame /= frames_for_bg_subtract.shape[0]

        max_blurred_frame -= mean_blurred_frame
        mean_normalization_value = np.max(max_blurred_frame)

        return mean_blurred_frame, mean_normalization_value

    def __len__(self):
        """Length of the dataset excluding the last frame."""
        return self.end_frame - self.start_frame - 1

    def __getitem__(self, idx: int):
        """Retrieve a batch of frames and labels."""
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        final_idx = min(idx + self.batch_size, len(self))
        frame_labels = None
        endpoint_capture = None

        if self.cache_bg_frames and all(
            i in self._frame_cache for i in range(idx, final_idx)
        ):
            cached = [self._frame_cache[i] for i in range(idx, final_idx)]
            frame_images = np.stack([item[0] for item in cached])
            unwarped_frames = (
                np.stack([item[1] for item in cached])
                if cached[0][1] is not None
                else None
            )
            echogram = (
                np.stack([item[2] for item in cached])
                if cached[0][2] is not None
                else None
            )

        else:

            frames, unwarped_frames = self.load_frames(
                self.start_frame + idx,
                self.start_frame + final_idx + 1,
                return_unwarped=self.return_unwarped or self.return_echogram,
            )
            if self.capture_endpoint_frames:
                frame_ids = self._loaded_frame_ids(self.start_frame + idx, len(frames))
                # A partial read/reset changes detector-frame alignment. Fail
                # explicitly instead of exporting endpoints under guessed IDs.
                expected_ids = tuple(
                    range(self.start_frame + idx, self.start_frame + final_idx + 1)
                )
                if frame_ids != expected_ids:
                    raise ValueError(
                        "Endpoint frame alignment changed during source decoding: "
                        f"expected {expected_ids}, received {frame_ids}"
                    )
            if self.return_unwarped:
                frame_images = (
                    self._stack_preprocessed_channels(
                        unwarped_frames,
                        self.unwarped_mean_blurred_frame,
                        self.unwarped_mean_normalization_value,
                    )
                    if self.do_bg_subtract
                    else np.expand_dims(unwarped_frames[:-1], -1)
                )
            else:
                frame_images = (
                    self._stack_preprocessed_channels(
                        frames,
                        self.mean_blurred_frame,
                        self.mean_normalization_value,
                    )
                    if self.do_bg_subtract
                    else np.expand_dims(frames[:-1], -1)
                )

            if self.capture_endpoint_frames:
                # Slice the existing native-resolution detector representation
                # before YOLO resizing. Retain only the middle channel.
                endpoint_capture = {
                    "frame_ids": frame_ids[:-1],
                    "images": tuple(
                        np.ascontiguousarray(image[..., 1]) for image in frame_images
                    ),
                    "preprocessing": self.endpoint_preprocessing,
                }

            # MAH 2025-02-07 17:13:36 Question, why are we removing the last frame? whether or not we are doing
            # background subtraction the image is 4D (previous behaviour was 4D for background subtracted was [t,h,w,
            # 3] not was [t,h,w])

            if self.return_unwarped or self.return_echogram:
                unwarped_frames = unwarped_frames[:-1]

            if self.return_echogram:
                echogram = self._get_echogram(unwarped_frames)
            else:
                echogram = None

            if self.cache_bg_frames:
                for position, image in enumerate(frame_images):
                    self._frame_cache[idx + position] = (
                        image,
                        (
                            unwarped_frames[position]
                            if unwarped_frames is not None
                            else None
                        ),
                        echogram[position] if echogram is not None else None,
                    )

        extra = (
            {"endpoint_capture": endpoint_capture}
            if self.capture_endpoint_frames
            else {}
        )
        return self._postprocess(
            frame_images,
            frame_labels,
            unwarped_frames,
            echogram,
            self.return_original_image,
            **extra,
        )

    def _loaded_frame_ids(self, requested_start, count):
        """Read source IDs exposed by the decoder, or contiguous IDs for simple loaders."""
        didson = getattr(self, "didson", None)
        frame_ids = getattr(didson, "last_loaded_frame_ids", None)
        if frame_ids is None:
            return tuple(range(requested_start, requested_start + count))
        if len(frame_ids) != count:
            raise ValueError("Decoder source frame IDs do not match loaded image count")
        return tuple(int(frame_id) for frame_id in frame_ids)

    def _stack_preprocessed_channels(
        self, frames: np.ndarray, mean_blurred_frame, mean_normalization_value
    ):
        """Generate a 3-channel representation of the frames.

        This method:
            • Applies Gaussian blurring to each frame (optionally with multithreading)
            • Normalizes blurred frames using either warped or unwarped mean statistics
            • Optionally clips negative normalized values for positive-only BGS
            • Scales normalized values into the [0, 255] range
            • Constructs a 3-channel image stack consisting of:
                - Channel 0: original frames (excluding the last)
                - Channel 1: blurred + normalized frames
                - Channel 2: temporal difference between consecutive blurred frames
        """
        T, H, W = frames.shape

        frame_image = np.empty((T - 1, H, W, 3), dtype=np.uint8)
        frame_image[..., 0] = frames[:-1]

        # Convert once
        frames_f32 = (
            frames.astype(np.float32, copy=False)
            if frames.dtype == np.float32
            else frames.astype(np.float32)
        )

        if not self.use_blur:
            blurred_frames = frames_f32  # reuse
        else:
            blurred_frames = np.empty_like(frames_f32)

            if self.use_multithreading:

                def worker(i):
                    # input is float32, output is float32
                    blurred_frames[i] = cv2.GaussianBlur(frames_f32[i], (5, 5), 0)

                run_with_threads(worker, list(range(T)), max_workers=self.max_workers)
            else:
                for i in range(T):
                    blurred_frames[i] = cv2.GaussianBlur(frames_f32[i], (5, 5), 0)

        # Normalize in-place (float32)
        blurred_frames -= mean_blurred_frame
        # Keep a fixed physical signal range when the sampled background is
        # constant, including in the detector/echogram preprocessing branch.
        normalization = (
            mean_normalization_value if mean_normalization_value > 0 else 255.0
        )
        blurred_frames /= normalization
        if self.only_pos_bgs:
            np.maximum(blurred_frames, 0, out=blurred_frames)
        else:
            blurred_frames = (blurred_frames + 1.0) * 0.5

        # Convert to uint8 channels
        # If you KNOW values are in [0,1], clip can be skipped; otherwise keep it.
        ch1 = np.clip(blurred_frames[:-1] * 255.0, 0, 255).astype(np.uint8)
        ch2 = np.clip(
            np.abs(blurred_frames[1:] - blurred_frames[:-1]) * 255.0, 0, 255
        ).astype(np.uint8)

        frame_image[..., 1] = ch1
        frame_image[..., 2] = ch2
        return frame_image

    def _postprocess(
        self,
        frame_images,
        frame_labels,
        unwarped_frames,
        echogram,
        return_original_image,
        endpoint_capture=None,
    ):
        """Postprocess frames before returning."""
        output = (
            frame_images,
            frame_labels,
            unwarped_frames,
            echogram,
            return_original_image,
        )
        return output + (endpoint_capture,) if endpoint_capture is not None else output

    def _get_echogram(self, unwarped_frames):
        """Generate Echogram from the frames.

        The return channels are the magnitude and the normalised angle between -0.5 and 0.5
        """
        unwarped_frames_bgs = unwarped_frames.astype(np.float32)
        if self.return_echogram_with_bg_subtracted:
            unwarped_frames_bgs -= self.unwarped_mean_blurred_frame
            unwarped_frames_bgs /= self.unwarped_mean_normalization_value
        echogram = np.max(unwarped_frames_bgs.astype(np.float32), axis=2).astype(
            np.float32
        )
        col = np.argmax(unwarped_frames_bgs, axis=2).astype(np.float32)
        col = col / unwarped_frames.shape[2]
        col -= 0.5
        echogram = np.stack([echogram, col], axis=2)

        return echogram

    def load_frames(self, idx, final_idx, return_unwarped):
        raise NotImplementedError("Subclasses should implement this method.")

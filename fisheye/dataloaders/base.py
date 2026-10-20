import cv2
import numpy as np
from torch.utils.data import Dataset

from fisheye.common.generic import run_with_threads
from fisheye.configs import BaseDatasetConfig
from fisheye.dataloaders.echogram import (
    compute_echogram,
    echogram_uses_bg_subtraction,
)
from fisheye.dataloaders.compute_bg_subtraction import compute_bg_subtraction


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
        self.extracted_frames = {}
        self.frame_labels = []
        self.extracted_unwarped_frames = {}
        self.extracted_echograms = {}
        self.independent_outputs = config.return_frames is not None
        self.legacy_primary_unwarped = (
            not self.independent_outputs and config.return_unwarped
        )
        self.return_frames = config.return_frames if self.independent_outputs else True
        self.legacy_raw_echogram = config._legacy_raw_echogram
        self.return_unwarped = config.return_unwarped or (
            not self.independent_outputs and config.return_echogram
        )
        self.return_echogram = config.return_echogram
        self.need_unwarped = self.return_unwarped or self.return_echogram
        self.echogram_channels = config.echogram_channels
        self.do_bg_subtract_echogram = echogram_uses_bg_subtraction(
            self.echogram_channels
        )
        self.max_workers = config.max_workers
        self.use_multithreading = config.use_multithreading
        self.use_blur = config.use_blur
        self.return_original_image = config.return_original_image

        if (
            not self.return_frames
            and not self.return_unwarped
            and not self.return_echogram
        ):
            raise ValueError(
                "At least one of return_frames, return_unwarped, or return_echogram "
                "must be True"
            )

        self._init_bg_frame()

    def _init_bg_frame(self):
        """Initialize background frame for subtraction."""
        need_warped_bg_subtract = (
            self.return_frames
            and not self.legacy_primary_unwarped
            and self.do_bg_subtract
        )
        need_unwarped_bg_subtract = (self.return_unwarped and self.do_bg_subtract) or (
            self.return_echogram and self.do_bg_subtract_echogram
        )

        if len(self) and (need_warped_bg_subtract or need_unwarped_bg_subtract):
            available = self.end_frame - self.start_frame
            # Preserve the trained image pipeline's background window. Echogram
            # statistics use exactly the requested count, independent of batch size.
            warped_count = (
                min(
                    available,
                    self.num_frames_bg_subtract // self.batch_size * self.batch_size
                    + 1,
                )
                if need_warped_bg_subtract
                else 0
            )
            unwarped_count = (
                min(
                    available,
                    (
                        self.num_frames_bg_subtract
                        if self.independent_outputs
                        else self.num_frames_bg_subtract
                        // self.batch_size
                        * self.batch_size
                        + 1
                    ),
                )
                if need_unwarped_bg_subtract
                else 0
            )
            num_frames_bg = max(warped_count, unwarped_count)
            frames_for_bg_subtract, unwarped_frames_for_bg_subtract = self.load_frames(
                self.start_frame,
                self.start_frame + num_frames_bg,
                return_unwarped=self.need_unwarped,
                return_warped=need_warped_bg_subtract,
            )

            if need_unwarped_bg_subtract:
                (
                    self.unwarped_mean_blurred_frame,
                    self.unwarped_mean_normalization_value,
                ) = self._compute_bg_subtraction(
                    unwarped_frames_for_bg_subtract[:unwarped_count]
                )

            if need_warped_bg_subtract:
                self.mean_blurred_frame, self.mean_normalization_value = (
                    self._compute_bg_subtraction(frames_for_bg_subtract[:warped_count])
                )

    def _compute_bg_subtraction(self, frames_for_bg_subtract):
        """Calculate the mean blurred frame and normalization value."""

        return compute_bg_subtraction(
            frames_for_bg_subtract,
            use_blur=self.use_blur,
            use_multithreading=self.use_multithreading,
            max_workers=self.max_workers,
        )

    def __len__(self):
        """Length of the dataset excluding the last frame."""
        return max(0, self.end_frame - self.start_frame - 1)

    def __getitem__(self, idx: int):
        """Retrieve a batch of frames and labels."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Frame index is outside the dataset")
        final_idx = min(idx + self.batch_size, len(self))
        frame_labels = None

        if self._is_cached(idx, final_idx):
            return self._postprocess(
                (
                    self._stack_cached(self.extracted_frames, idx, final_idx)
                    if self.return_frames
                    else None
                ),
                frame_labels,
                (
                    self._stack_cached(self.extracted_unwarped_frames, idx, final_idx)
                    if self.return_unwarped
                    else None
                ),
                (
                    self._stack_cached(self.extracted_echograms, idx, final_idx)
                    if self.return_echogram
                    else None
                ),
                self.return_original_image if self.return_frames else False,
            )

        else:
            frames, unwarped_frames = self.load_frames(
                self.start_frame + idx,
                self.start_frame + final_idx + 1,
                return_unwarped=self.need_unwarped,
                return_warped=self.return_frames and not self.legacy_primary_unwarped,
            )

            frame_images = None
            if self.return_frames:
                primary_frames = (
                    unwarped_frames if self.legacy_primary_unwarped else frames
                )
                mean_prefix = "unwarped_" if self.legacy_primary_unwarped else ""
                frame_images = (
                    self._stack_preprocessed_channels(
                        primary_frames,
                        mean_blurred_frame=getattr(
                            self, mean_prefix + "mean_blurred_frame"
                        ),
                        mean_normalization_value=getattr(
                            self, mean_prefix + "mean_normalization_value"
                        ),
                    )
                    if self.do_bg_subtract
                    else np.expand_dims(primary_frames[:-1], -1)
                )

            raw_unwarped_frames = unwarped_frames[:-1] if self.need_unwarped else None

            returned_unwarped_frames = None
            if self.return_unwarped:
                returned_unwarped_frames = (
                    self._stack_preprocessed_channels(
                        unwarped_frames,
                        mean_blurred_frame=self.unwarped_mean_blurred_frame,
                        mean_normalization_value=self.unwarped_mean_normalization_value,
                    )
                    if self.do_bg_subtract and self.independent_outputs
                    else raw_unwarped_frames
                )

            echogram = (
                self._get_echogram(raw_unwarped_frames)
                if self.return_echogram
                else None
            )

            if self.cache_bg_frames:
                if frame_images is not None:
                    self.extracted_frames.update(enumerate(frame_images, start=idx))
                self.frame_labels = None
                if returned_unwarped_frames is not None:
                    self.extracted_unwarped_frames.update(
                        enumerate(returned_unwarped_frames, start=idx)
                    )
                if echogram is not None:
                    self.extracted_echograms.update(enumerate(echogram, start=idx))

        return self._postprocess(
            frame_images,
            frame_labels,
            returned_unwarped_frames,
            echogram,
            self.return_original_image if self.return_frames else False,
        )

    def _stack_preprocessed_channels(
        self, frames: np.ndarray, mean_blurred_frame, mean_normalization_value
    ):
        """Generate a 3-channel representation of the frames.

        This method:
            • Applies Gaussian blurring to each frame (optionally with multithreading)
            • Normalizes blurred frames using either warped or unwarped mean statistics
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
        blurred_frames /= mean_normalization_value
        blurred_frames = (blurred_frames + 1.0) * 0.5  # combine two ops

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
    ):
        """Postprocess frames before returning."""
        return (
            frame_images,
            frame_labels,
            unwarped_frames,
            echogram,
            return_original_image,
        )

    def _get_echogram(self, unwarped_frames):
        """Generate echogram from unwarped frames (delegates to shared compute_echogram)."""
        echogram = compute_echogram(
            unwarped_frames,
            mean_blurred_frame=getattr(self, "unwarped_mean_blurred_frame", None),
            mean_normalization_value=getattr(
                self, "unwarped_mean_normalization_value", None
            ),
            echogram_channels=self.echogram_channels,
        )
        if self.legacy_raw_echogram:
            echogram[..., 0] = np.max(unwarped_frames, axis=2)
        return echogram

    def _is_cached(self, idx, final_idx):
        """Only reuse a batch when each requested frame is cached."""
        caches = []
        if self.return_frames:
            caches.append(self.extracted_frames)
        if self.return_unwarped:
            caches.append(self.extracted_unwarped_frames)
        if self.return_echogram:
            caches.append(self.extracted_echograms)
        return bool(caches) and all(
            frame_idx in cache
            for cache in caches
            for frame_idx in range(idx, final_idx)
        )

    @staticmethod
    def _stack_cached(cache, idx, final_idx):
        return np.stack([cache[frame_idx] for frame_idx in range(idx, final_idx)])

    def load_frames(self, idx, final_idx, return_unwarped=False, return_warped=True):
        raise NotImplementedError("Subclasses should implement this method.")

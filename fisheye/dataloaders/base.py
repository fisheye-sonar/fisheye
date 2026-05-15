import cv2
import numpy as np
from torch.utils.data import Dataset

from fisheye.common.generic import run_with_threads
from fisheye.configs import BaseDatasetConfig
from fisheye.dataloaders.echogram import compute_bg_subtraction, compute_echogram


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
        self.return_echogram_with_bg_subtracted = (
            config.return_echogram_with_bg_subtracted
        )

        self.return_echogram_with_how_wide_the_peak_as_third_channel = (
            config.return_echogram_with_how_wide_the_peak_as_third_channel
        )
        self.return_echogram_with_no_bgs_as_third_channel = (
            config.return_echogram_with_no_bgs_as_third_channel
        )
        self.return_echogram_with_distances_as_third_channel = (
            config.return_echogram_with_distances_as_third_channel
        )
        if self.return_echogram_with_distances_as_third_channel:
            print(
                "return_echogram_with_distances_as_third_channel is True, but will only return zeros to be post processed to add in"
            )
        self.extracted_frames = []
        self.frame_labels = []
        self.extracted_unwarped_frames = []
        self.extracted_echograms = []
        self.return_unwarped = config.return_unwarped
        self.return_echogram = config.return_echogram
        self.only_echogram = config.only_echogram
        if self.only_echogram and not self.return_echogram:
            raise ValueError("only_echogram requires return_echogram=True")
        self.max_workers = config.max_workers
        self.use_multithreading = config.use_multithreading
        self.use_blur = config.use_blur
        self.return_original_image = config.return_original_image

        self._init_bg_frame()

    def _init_bg_frame(self):
        """Initialize background frame for subtraction."""
        need_unwarped = (
            self.return_unwarped or self.return_echogram or self.only_echogram
        )
        if self.do_bg_subtract or self.return_echogram:
            num_frames_bg = min(
                self.end_frame - self.start_frame,
                self.num_frames_bg_subtract // self.batch_size * self.batch_size + 1,
            )
            frames_for_bg_subtract, unwarped_frames_for_bg_subtract = self.load_frames(
                self.start_frame,
                self.start_frame + num_frames_bg,
                return_unwarped=need_unwarped,
                skip_warped=self.only_echogram or not self.do_bg_subtract,
            )

            if need_unwarped:
                (
                    self.unwarped_mean_blurred_frame,
                    self.unwarped_mean_normalization_value,
                ) = self._compute_bg_subtraction(unwarped_frames_for_bg_subtract)

        if self.do_bg_subtract and not self.only_echogram:
            self.mean_blurred_frame, self.mean_normalization_value = (
                self._compute_bg_subtraction(frames_for_bg_subtract)
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
        return self.end_frame - self.start_frame - 1

    def __getitem__(self, idx: int):
        """Retrieve a batch of frames and labels."""
        final_idx = min(idx + self.batch_size, len(self))
        frame_labels = None

        if idx + 1 < len(self.extracted_frames):
            return self._postprocess(
                np.stack(self.extracted_frames[idx:final_idx]),
                frame_labels,
                np.stack(self.extracted_unwarped_frames[idx:final_idx]),
                np.stack(self.extracted_echograms[idx:final_idx]),
                self.return_original_image,
            )

        else:
            need_unwarped = (
                self.return_unwarped or self.return_echogram or self.only_echogram
            )
            frames, unwarped_frames = self.load_frames(
                self.start_frame + idx,
                self.start_frame + final_idx + 1,
                return_unwarped=need_unwarped,
                skip_warped=self.only_echogram,
            )

            if self.only_echogram:
                frame_images = None
                unwarped_frames = unwarped_frames[:-1]
                echogram = self._get_echogram(unwarped_frames)
            else:
                if self.return_unwarped:
                    frame_images = unwarped_frames
                else:
                    frame_images = frames

                # MAH 2025-02-07 17:13:36 Question, why are we removing the last frame? whether or not we are doing
                # background subtraction the image is 4D (previous behaviour was 4D for background subtracted was [t,h,w,
                # 3] not was [t,h,w])
                frame_images = (
                    self._stack_preprocessed_channels(frame_images)
                    if self.do_bg_subtract
                    else np.expand_dims(frame_images[:-1], -1)
                )
                if need_unwarped:
                    unwarped_frames = unwarped_frames[:-1]

                if self.return_echogram:
                    echogram = self._get_echogram(unwarped_frames)
                else:
                    echogram = None

            if self.cache_bg_frames:
                if frame_images is not None:
                    self.extracted_frames.extend(frame_images)
                self.frame_labels = None
                if not self.only_echogram:
                    self.extracted_unwarped_frames.extend(unwarped_frames)
                self.extracted_echograms.extend(echogram)

        return self._postprocess(
            frame_images,
            frame_labels,
            None if self.only_echogram else unwarped_frames,
            echogram,
            False if self.only_echogram else self.return_original_image,
        )

    def _stack_preprocessed_channels(self, frames: np.ndarray):
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
        if not self.use_blur:
            blurred_frames = frames.astype(np.float32)

        else:
            if self.use_multithreading:
                blurred_frames = np.zeros_like(frames, dtype=np.float32)
                blurred_frames_list = run_with_threads(
                    lambda i: cv2.GaussianBlur(frames[i], (5, 5), 0),
                    list(range(frames.shape[0])),
                    max_workers=self.max_workers,
                )

                for i in range(frames.shape[0]):
                    blurred_frames[i] = blurred_frames_list[i]
            else:
                blurred_frames = frames.astype(np.float32)
                for i in range(frames.shape[0]):
                    blurred_frames[i] = cv2.GaussianBlur(blurred_frames[i], (5, 5), 0)

        if self.return_unwarped:
            blurred_frames -= self.unwarped_mean_blurred_frame
            blurred_frames /= self.unwarped_mean_normalization_value
        else:
            blurred_frames -= self.mean_blurred_frame
            blurred_frames /= self.mean_normalization_value

        # MAH 2025-11-24 17:09:09 I think we should not do this here and instead we should only take the positive values of the bgs
        blurred_frames += 1
        blurred_frames /= 2

        frame_image = np.stack(
            [
                frames[:-1],
                blurred_frames[:-1] * 255,
                np.abs(blurred_frames[1:] - blurred_frames[:-1]) * 255,
            ],
            axis=-1,
        ).astype(np.uint8, copy=False)

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
        return compute_echogram(
            unwarped_frames,
            mean_blurred_frame=getattr(self, "unwarped_mean_blurred_frame", None),
            mean_normalization_value=getattr(
                self, "unwarped_mean_normalization_value", None
            ),
            return_echogram_with_bg_subtracted=self.return_echogram_with_bg_subtracted,
            return_echogram_with_how_wide_the_peak_as_third_channel=self.return_echogram_with_how_wide_the_peak_as_third_channel,
            return_echogram_with_no_bgs_as_third_channel=self.return_echogram_with_no_bgs_as_third_channel,
            return_echogram_with_distances_as_third_channel=self.return_echogram_with_distances_as_third_channel,
        )

    def load_frames(self, idx, final_idx, return_unwarped):
        raise NotImplementedError("Subclasses should implement this method.")

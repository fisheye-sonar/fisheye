import structlog

from fisheye.configs import BaseDatasetConfig
from fisheye.configs.datasets import ARISMetadata, BEAM_WIDTH_DIR
from fisheye.dataloaders.base import BaseDataset
from fisheye.dataloaders.didson.pyDIDSON import DIDSON, compute_resized_shape

logger = structlog.get_logger()


class ARISBatchedDataset(BaseDataset):
    """ARISBatchedDataset

    A PyTorch Dataset for loading and preprocessing frames from ARIS/DIDSON files. This includes frame extraction,
    optional background subtraction, and batching.
    """

    def __init__(self, config: BaseDatasetConfig):
        """
        Initialize the ARISBatchedDataset with configuration options.

        Args:
            config (BaseDatasetConfig): Configuration object containing all dataset parameters.
        """
        try:
            self.didson = DIDSON(
                config.filepath,
                beam_width_dir=BEAM_WIDTH_DIR,
                img_load_size=config.img_load_size,
                img_size=config.img_size,
                stride=config.stride,
                return_original_image=config.return_original_image,
            )
        except Exception as e:
            logger.error(
                "failed_to_load_file",
                filepath=config.filepath,
                error=str(e),
                exc_info=True,
            )

            raise RuntimeError(f"Could not load {config.filepath}") from e

        config.start_frame, config.end_frame = self._validate_frame_range(config=config)
        self.metadata = self._extract_metadata()

        super().__init__(config)

    def _extract_metadata(self) -> ARISMetadata:
        info = self.didson.info
        return ARISMetadata(
            xdim=info.get("xdim", 0),
            ydim=info.get("ydim", 0),
            image_meter_width=info["xdim"] * info["pixel_meter_width"],
            image_meter_height=info["ydim"] * info["pixel_meter_height"],
            pixel_meter_size=info.get("pixel_meter_size", 0),
            x_meter_start=info.get("x_meter_start", 0),
            x_meter_stop=info.get("x_meter_stop", 0),
            y_meter_start=info.get("y_meter_start", 0),
            y_meter_stop=info.get("y_meter_stop", 0),
            sampleperiod=info.get("sampleperiod", 0),
            soundspeed=info.get("soundspeed", 0),
            windowstart=info.get("windowstart", 0),
            samplesperbeam=info.get("samplesperbeam", 0),
            BeamCount=info.get("BeamCount", 0),
            thesystemtype=info.get("thesystemtype", 0),
            largelens=info.get("largelens", 0),
            numframes=info.get("numframes", 0),
            unwarped_shape=info.get("unwarped_shape", (0, 0)),
            beam_width_data=info.get("beam_width_data", None),
        )

    def load_frames(self, start_frame, end_frame, return_unwarped=False):
        """Load ARIS frames."""
        return self.didson.load_frames(
            start_frame=start_frame,
            end_frame=end_frame,
            return_unwarped=return_unwarped,
        )

    def _validate_frame_range(self, config):
        """Validate the start and end frame IDs."""
        end_frame = self.didson.info["numframes"]

        config.end_frame = (
            end_frame
            if not config.end_frame or config.end_frame > end_frame
            else config.end_frame
        )

        # TODO (MVH) - this code block could be removed since we covered this edge case on line 56. However,
        #  since it's been an issue a few, unexpected times, we will keep it here a little longer.
        if config.end_frame <= 0:
            # If end frame is 0 or -1, something ain't right in the header file. However, there most likely is still
            # data that can be unpacked so load all frames.
            config.end_frame = end_frame
            config.start_frame, config.end_frame = self._validate_frame_range(
                config=config
            )

        # We are possibly looking at a shortened clip where the start and end frame indexes are larger than the number
        # of frames in the file.
        if config.start_frame > config.end_frame:
            logger.warning(
                "End frame is 0 or -1, likely due to a corrupted or incomplete header file. "
                "Even if you provided a valid end_frame, it was overwritten because the original end_frame is smaller. "
                "Falling back to loading all frames, which may be inefficient."
            )
            # Reset the start and end frames
            config.start_frame = 0
            config.end_frame = end_frame
            logger.warning(
                f"Resetting start_frame to 0 and end_frame to {config.end_frame}."
            )

        return config.start_frame, config.end_frame

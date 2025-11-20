"""High-level dataset builder that orchestrates the dataset creation and export process."""

import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from fisheye.dataset.enums import DatasetFormat
from fisheye.dataset.parser import parse_aris_xml, find_matching_aris_xml_files
from fisheye.configs.datasets import BaseDatasetConfig
from fisheye.dataloaders import create_dataloader
from fisheye.dataloaders.didson.pyDIDSON import DIDSON
from fisheye.export.dataset import DATASET_EXPORTER_REGISTRY, BaseExporter

logger = logging.getLogger(__name__)


def get_bbox_with_padding(coords_px, padding, min_padding_px, frame_shape):
    """Get bounding box coordinates with padding."""
    x_min = int(coords_px[:, 0].min())
    x_max = int(coords_px[:, 0].max())
    y_min = int(coords_px[:, 1].min())
    y_max = int(coords_px[:, 1].max())

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_pad = max(int(np.ceil(padding * max(1, x_range))), min_padding_px)
    y_pad = max(int(np.ceil(padding * max(1, y_range))), min_padding_px)

    x_start = max(x_min - x_pad, 0)
    y_start = max(y_min - y_pad, 0)
    x_stop = min(x_max + x_pad + 1, frame_shape[-1])
    y_stop = min(y_max + y_pad + 1, frame_shape[-2])

    return x_start, y_start, x_stop, y_stop


def coords_meters_to_pixels(coords: np.ndarray, metadata: dict):
    """Convert ARIS world coords to pixel coords."""
    x_aris_max = metadata["x_meter_stop"]
    x_aris_min = metadata["x_meter_start"]
    y_aris_max = metadata["y_meter_start"]
    y_aris_min = metadata["y_meter_stop"]

    xdim = int(metadata["xdim"])
    ydim = int(metadata["ydim"])

    # Convert to pixel space
    x = (xdim) * (coords[:, 0] - x_aris_min) / (x_aris_max - x_aris_min)
    y = (ydim) * (y_aris_max - coords[:, 1]) / (y_aris_max - y_aris_min)

    # Clip to valid pixel ranges
    coords_px = np.rint(np.stack((x, y), axis=1)).astype(np.int64)
    coords_px[:, 0] = np.clip(coords_px[:, 0], 0, xdim - 1)
    coords_px[:, 1] = np.clip(coords_px[:, 1], 0, ydim - 1)

    return coords_px


def get_box_data_from_xml(fish_data, metadata, padding: float, min_padding_px: int):
    """Extracts bounding box data (ARIS world coords) from fish data in XML format."""
    all_bbox_data = []
    for entry in fish_data:
        frame_idx = int(entry["@FrameIndex"])
        fish_id = int(entry["@FishID"])
        coords = _parse_nodes(entry["FishMeasureNode"])
        coords_px = coords_meters_to_pixels(coords, metadata)
        x_start, y_start, x_stop, y_stop = get_bbox_with_padding(
            coords_px,
            padding,
            min_padding_px,
            (int(metadata["ydim"]), int(metadata["xdim"])),
        )
        all_bbox_data.append(
            {
                "frame_idx": frame_idx,
                "fish_id": fish_id,
                "bbox_xy_xy": [x_start, y_start, x_stop, y_stop],
                "fish_coords_xyxy": coords_px,
                "fish_coords_meters": coords,
            }
        )

    return all_bbox_data


def _parse_nodes(nodes):
    # nodes might be list or dict depending on xml shape
    if isinstance(nodes, dict):
        nodes = [nodes]
    arr = np.array(
        [[float(n["@WorldPointX"]), float(n["@WorldPointY"])] for n in nodes]
    )

    return arr


class DatasetBuilder:
    """High-level dataset builder that orchestrates the dataset creation and export process."""

    def __init__(
        self,
        aris_dir: Union[Path, str],
        xml_dir: Union[Path, str],
        out_dir: Union[Path, str],
        dataset_format: DatasetFormat = DatasetFormat.YOLO,
        padding: float = 0.1,
        min_padding_px: int = 30,
    ):
        """Initializes the dataset builder."""
        self.aris_dir = Path(aris_dir)
        self.xml_dir = Path(xml_dir)
        self.out_dir = Path(out_dir)
        self.padding = padding
        self.min_padding_px = min_padding_px
        self.images_dir = self.out_dir / "images"
        self.annotations_dir = self.out_dir / "annotations"
        for p in [self.images_dir, self.annotations_dir]:
            p.mkdir(parents=True, exist_ok=True)

        self.exporter: BaseExporter = DATASET_EXPORTER_REGISTRY[dataset_format](
            self.annotations_dir
        )

    def __call__(self):
        """Main entry point for dataset creation and export."""
        aris_paths, xml_paths, unpaired_xml, unpaired_aris = (
            find_matching_aris_xml_files(self.aris_dir, self.xml_dir)
        )
        logger.info("Found %d pairs to process", len(aris_paths))
        for aris_pth, xml_pth in zip(aris_paths, xml_paths):
            try:
                self._process_pair(aris_pth, xml_pth)
            except Exception as e:
                logger.exception("Error processing %s / %s: %s", aris_pth, xml_pth, e)

    def _process_pair(self, aris_pth: Path, xml_pth: Path):
        """Processes a pair of ARIS and XML files."""
        metadata = DIDSON(aris_pth).info
        # Retrieve ARISFish's MarkedFishMeasurements node
        xml_output = parse_aris_xml(xml_pth)
        if not xml_output:
            logger.debug("No fish data in %s", xml_pth)
            return

        mfm_data = xml_output.get("MarkedFishMeasurement")
        if isinstance(mfm_data, dict):
            mfm_data = [mfm_data]

        bbox_data_list = get_box_data_from_xml(
            mfm_data, metadata, self.padding, self.min_padding_px
        )

        for bbox_data in bbox_data_list:
            frame_idx = bbox_data["frame_idx"]
            aris_stem = Path(aris_pth).stem

            config = BaseDatasetConfig(
                filepath=str(aris_pth),
                start_frame=frame_idx,
                end_frame=frame_idx
                + 2,  # Include extra frames: 1 for background subtraction, 1 because the last frame is usually skipped
            )

            dataloader, _ = create_dataloader(config)

            for images, _, _, _ in dataloader:
                for image in images:
                    im = Image.fromarray(image.numpy())
                    im.save(
                        self.images_dir / f"{aris_stem}_{frame_idx:06d}.jpg", quality=95
                    )

            out_ann_file = f"{aris_stem}_{frame_idx:06d}"
            self.exporter(bbox_data, out_ann_file, metadata)


builder = DatasetBuilder(
    aris_dir="/Users/madison/Downloads/Klamath_2024_xml_sample/",
    xml_dir="/Users/madison/Downloads/Klamath_2024_xml_sample/",
    out_dir="/Users/madison/Downloads/Klamath_2024_xml_sample/test_out",
    padding=0.1,
    min_padding_px=30,
)
builder()

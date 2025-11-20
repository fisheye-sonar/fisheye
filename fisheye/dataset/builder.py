"""High-level dataset builder that orchestrates the dataset creation and export process."""

from pathlib import Path
from typing import Union, Iterable, Tuple

import structlog
from PIL import Image

from fisheye.dataloaders.didson.pyDIDSON import DIDSON
from fisheye.dataloaders.utils import FrameExtractor
from fisheye.dataset.enums import DatasetFormat
from fisheye.dataset.parser import (
    parse_aris_xml,
    find_matching_aris_xml_files,
    get_box_data_from_xml,
)
from fisheye.export.dataset import DATASET_EXPORTER_REGISTRY, BaseExporter

logger = structlog.getLogger(__name__)


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
        extra_frames: int = 2,
        frame_extractor: FrameExtractor | None = None,
        exporter: BaseExporter | None = None,
    ):
        """Initializes the dataset builder."""
        self.aris_dir = Path(aris_dir)
        self.xml_dir = Path(xml_dir)
        self.out_dir = Path(out_dir)
        self.padding = padding
        self.min_padding_px = min_padding_px
        self.images_dir = self.out_dir / "images"
        self.annotations_dir = self.out_dir / dataset_format.value.lower()
        for p in [self.images_dir, self.annotations_dir]:
            p.mkdir(parents=True, exist_ok=True)

        self.frame_extractor = frame_extractor or FrameExtractor(
            extra_frames=extra_frames
        )
        self.exporter = exporter or DATASET_EXPORTER_REGISTRY[dataset_format](
            self.annotations_dir
        )

    def build_all(self):
        """Main entry point for dataset creation and export."""
        aris_paths, xml_paths, unpaired_xml, unpaired_aris = (
            find_matching_aris_xml_files(self.aris_dir, self.xml_dir)
        )

        if unpaired_xml:
            logger.warning("unpaired_xml_files_found", count=len(unpaired_xml))
        if unpaired_aris:
            logger.warning("unpaired_aris_files_found", count=len(unpaired_aris))

        logger.info("paired_files_found", count=len(aris_paths))

        self.build_from_pairs(zip(aris_paths, xml_paths))

    __call__ = build_all

    def build_from_pairs(self, pairs: Iterable[Tuple[Path, Path]]):
        """Build dataset for a provided iterable of (aris_path, xml_path) pairs."""
        for aris_pth, xml_pth in pairs:
            try:
                self._process_pair(aris_pth, xml_pth)
            except Exception as e:
                logger.error(
                    "dataset_creation_error",
                    aris=str(aris_pth),
                    xml=str(xml_pth),
                    error=str(e),
                )

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

        aris_stem = Path(aris_pth).stem
        for bbox_data in bbox_data_list:
            frame_idx = bbox_data["frame_idx"]
            self._save_frames_to_disk(aris_pth, aris_stem, frame_idx)
            self._export_annotations(aris_stem, frame_idx, bbox_data, metadata)

    def _save_frames_to_disk(self, aris_path: Path, aris_stem: str, frame_idx: int):
        """Extract and save images for a specific bbox frame index."""
        for image_tensor in self.frame_extractor.iter_frames(aris_path, frame_idx):
            im = Image.fromarray(image_tensor.numpy())
            im.save(self.images_dir / f"{aris_stem}_{frame_idx:06d}.jpg", quality=95)

    def _export_annotations(
        self,
        aris_stem: str,
        frame_idx: int,
        bbox_data: dict,
        metadata: dict,
    ):
        """Export annotations for a frame."""
        out_ann_file = f"{aris_stem}_{frame_idx:06d}"
        self.exporter(bbox_data, out_ann_file, metadata)

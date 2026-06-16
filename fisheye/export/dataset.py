from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Type, Union

from fisheye.dataset.enums import DatasetFormat
from fisheye.boxes import convert_xyxy_to_cxcywh

import numpy as np
import json


class BaseDatasetExporter(ABC):
    """Abstract base class for dataset exporters."""

    def __call__(self, bbox_data, img_name, metadata):
        return self.export(bbox_data, img_name, metadata)

    @abstractmethod
    def export(
        self,
        bbox_data: Dict[str, Any],
        img_name: Union[Path, str],
        metadata: Dict[str, Any],
    ) -> None:
        pass


class YOLOExporter(BaseDatasetExporter):
    """YOLO dataset exporter."""

    def __init__(self, annotations_dir: Path):
        self.annotations_dir = annotations_dir

    def export(
        self,
        bbox_data: Dict[str, Any],
        img_name: Union[Path, str],
        metadata: Dict[str, Any],
    ):
        bbox_cxcywh = convert_xyxy_to_cxcywh(
            bbox_data["bbox_xy_xy"], metadata["xdim"], metadata["ydim"]
        )
        out_file = self.annotations_dir / f"{img_name}.txt"
        with open(out_file, "w") as f:
            f.write(
                f"0 {bbox_cxcywh[0]} {bbox_cxcywh[1]} {bbox_cxcywh[2]} {bbox_cxcywh[3]}\n"
            )


class COCOExporter(BaseDatasetExporter):
    """COCO dataset exporter."""

    def __init__(self, annotations_dir: Path):
        self.annotations_dir = annotations_dir

    def export(
        self,
        bbox_data: Dict[str, Any],
        img_name: Union[Path, str],
        metadata: Dict[str, Any],
    ):
        raise NotImplementedError("COCO export not yet implemented")


class FrameFishInstanceAnnotationExporter(BaseDatasetExporter):
    """Export one JSON annotation for a single fish instance in a frame.

    The output preserves the source fish geometry, includes relevant metadata
    from the ARIS file, and adds derived fields used by downstream tasks such
    as bounding-box and length training.
    """

    def __init__(self, annotations_dir: Path):
        self.annotations_dir = annotations_dir

    def export(
        self,
        bbox_data: Dict[str, Any],
        img_name: Union[Path, str],
        metadata: Dict[str, Any],
    ):
        for key, value in bbox_data.items():
            if isinstance(value, np.ndarray):
                bbox_data[key] = value.tolist()
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        value[k] = v.tolist()
        bbox_data = {
            k: v for k, v in bbox_data.items() if not isinstance(v, np.ndarray)
        }
        out_file = self.annotations_dir / f"{img_name}.json"
        bbox_data["bbox_yolo"] = convert_xyxy_to_cxcywh(
            bbox_data["bbox_xy_xy"], metadata["xdim"], metadata["ydim"]
        )
        bbox_data["pixel_meter_size"] = metadata["pixel_meter_size"]
        bbox_data["xdim"] = metadata["xdim"]
        bbox_data["ydim"] = metadata["ydim"]
        xml_xyxy_met = bbox_data["fish_coords_meters"]
        xml_range_0 = (
            (xml_xyxy_met[0][0]) ** 2 + ((xml_xyxy_met[0][1]) ** 2)
        ) ** 0.5  # range to head as saved by arisfish
        xml_range_1 = (
            (xml_xyxy_met[1][0]) ** 2 + ((xml_xyxy_met[1][1]) ** 2)
        ) ** 0.5  # range to tail as saved by arisfish

        xml_length = (
            (xml_xyxy_met[0][0] - xml_xyxy_met[-1][0]) ** 2
            + (xml_xyxy_met[0][1] - xml_xyxy_met[-1][1]) ** 2
        ) ** 0.5
        bbox_data["xml_range"] = (xml_range_0 + xml_range_1) / 2
        bbox_data["xml_length"] = xml_length
        with open(out_file, "w") as f:
            json.dump(bbox_data, f)


DATASET_EXPORTER_REGISTRY: Dict[DatasetFormat, Type[BaseDatasetExporter]] = {
    DatasetFormat.YOLO: YOLOExporter,
    DatasetFormat.COCO: COCOExporter,
    DatasetFormat.SFFI: FrameFishInstanceAnnotationExporter,
}

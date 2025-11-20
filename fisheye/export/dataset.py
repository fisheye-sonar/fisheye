from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Type

from fisheye.dataset.enums import DatasetFormat
from fisheye.boxes import convert_xyxy_to_cxcywh


class BaseExporter(ABC):
    """Abstract base class for dataset exporters."""

    def __call__(self, bbox_data, img_name, metadata):
        return self.export(bbox_data, img_name, metadata)

    @abstractmethod
    def export(self, bbox_data: Dict[str, Any], img_name: str, metadata: Dict) -> None:
        pass


class YOLOExporter(BaseExporter):
    """YOLO dataset exporter."""

    def __init__(self, annotations_dir: Path):
        self.annotations_dir = annotations_dir

    def export(self, bbox_data, img_name, metadata):
        bbox_cxcywh = convert_xyxy_to_cxcywh(
            bbox_data["bbox_xy_xy"], metadata["xdim"], metadata["ydim"]
        )
        out_file = self.annotations_dir / f"{img_name}.txt"
        with open(out_file, "w") as f:
            f.write(
                f"0 {bbox_cxcywh[0]} {bbox_cxcywh[1]} {bbox_cxcywh[2]} {bbox_cxcywh[3]}\n"
            )


class COCOExporter(BaseExporter):
    """COCO dataset exporter."""

    def export(self, bbox_data, img_name, metadata):
        raise NotImplementedError("COCO export not yet implemented")


DATASET_EXPORTER_REGISTRY: Dict[DatasetFormat, Type[BaseExporter]] = {
    DatasetFormat.YOLO: YOLOExporter,
    DatasetFormat.COCO: COCOExporter,
}

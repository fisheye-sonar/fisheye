import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from fisheye.common.file_system import find_aris_xml_files
from fisheye.dataset.utils import coords_meters_to_pixels, get_bbox_with_padding


def etree_to_dict(el: ET.Element) -> Dict[str, Any]:
    """Convert an XML element tree to a dictionary."""
    d = {el.tag: {} if el.attrib or list(el) else None}
    children = list(el)
    if children:
        dd = {}
        for ch in children:
            cd = etree_to_dict(ch)
            for k, v in cd.items():
                if k in dd:
                    if not isinstance(dd[k], list):
                        dd[k] = [dd[k]]
                    dd[k].append(v)
                else:
                    dd[k] = v
        d[el.tag] = dd
    if el.attrib:
        d[el.tag].update({f"@{k}": v for k, v in el.attrib.items()})
    text = (el.text or "").strip()
    if text:
        if children or el.attrib:
            d[el.tag]["#text"] = text
        else:
            d[el.tag] = text

    return d


def parse_aris_xml(path: Path):
    """Parse ARISFish's XML output and return the MarkedFishMeasurements node or None."""
    with open(path, "r") as f:
        root = ET.parse(f).getroot()

    data = etree_to_dict(root)

    return data.get("MarkedFishMeasurements")


def find_matching_aris_xml_files(aris_dir: Union[Path, str], xml_dir: Union[Path, str]):
    """Find ARIS files and their corresponding XML files.

    Matching pairs are required for building and exporting the dataset.
    """
    aris_dir = Path(aris_dir)
    xml_dir = Path(xml_dir)

    aris_paths: List[Path] = [Path(p) for p in find_aris_xml_files(aris_dir, "*.aris")]
    xml_paths: List[Path] = [Path(p) for p in find_aris_xml_files(xml_dir, "*.xml")]

    # XML file names: paths
    xml_by_name: Dict[str, Path] = {fp.name: fp for fp in xml_paths}

    aris_paths_with_xml = []
    xml_paths_with_aris = []

    for aris_path in aris_paths:
        file_name = aris_path.stem
        predicted_xml_fn = f"FCe_{file_name}_ID_.xml"

        xml_path = xml_by_name.get(predicted_xml_fn)

        if xml_path is not None:
            aris_paths_with_xml.append(aris_path)
            xml_paths_with_aris.append(xml_path)

    unpaired_xml_files = [fp for fp in xml_paths if fp not in xml_paths_with_aris]
    unpaired_aris_files = [fp for fp in aris_paths if fp not in aris_paths_with_xml]

    aris_paths, xml_paths = aris_paths_with_xml, xml_paths_with_aris

    return aris_paths, xml_paths, unpaired_xml_files, unpaired_aris_files


def get_box_data_from_xml(
    fish_data: dict, metadata: dict, padding: float, min_padding_px: int
):
    """Extracts bounding box data (ARIS world coords) from fish data in XML format.

    Returns a list of per-fish dictionaries with:
      - frame_idx
      - fish_id
      - bbox_xy_xy
      - fish_coords_xyxy
      - fish_coords_meters
    """
    all_bbox_data: List[Dict[str, Any]] = []

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


def _parse_nodes(nodes: Union[List, Dict]):
    """Parse FishMeasureNode elements into an (N, 2) array of world coordinates.

    `nodes` may be a dict or a list of dicts depending on the XML shape.
    """
    if isinstance(nodes, dict):
        nodes = [nodes]
    coords = [(float(n["@WorldPointX"]), float(n["@WorldPointY"])) for n in nodes]
    return np.array(coords, dtype=float)

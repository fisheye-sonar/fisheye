import xml.etree.ElementTree as ET
from typing import Any, Dict
from pathlib import Path

from fisheye.common.file_system import find_real_files


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


def find_matching_aris_xml_files(aris_dir, xml_dir):
    """Find ARIS files and their corresponding XML files.

    Matching pairs are required for building and exporting the dataset.
    """
    aris_p = Path(aris_dir)
    xml_p = Path(xml_dir)
    aris_paths = find_real_files(aris_p, "*.aris")
    xml_paths = find_real_files(xml_p, "*.xml")
    xml_fns = [fp.name for fp in xml_paths]

    aris_paths_with_xml = []
    xml_paths_with_aris = []

    for aris_path in map(Path, aris_paths):
        file_name = aris_path.stem
        predicted_xml_fn = f"FCe_{file_name}_ID_.xml"

        if predicted_xml_fn in xml_fns:
            # get the index of the predicted_xml_fn in xml_fns
            index = xml_fns.index(predicted_xml_fn)
            aris_paths_with_xml.append(str(aris_path))
            xml_paths_with_aris.append(str(xml_paths[index]))

    unpaired_xml_files = [fp for fp in xml_paths if fp not in xml_paths_with_aris]
    unpaired_aris_files = [fp for fp in aris_paths if fp not in aris_paths_with_xml]

    aris_paths, xml_paths = aris_paths_with_xml, xml_paths_with_aris

    return aris_paths, xml_paths, unpaired_xml_files, unpaired_aris_files

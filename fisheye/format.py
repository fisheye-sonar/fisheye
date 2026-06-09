from typing import Union, Dict, List

from fisheye.configs.datasets import ARISMetadata
from fisheye.export import FC_SCHEMA


def tracker_output_to_dict_rows(data: Dict):
    """Convert tracker output containing bounding boxes from original image space in [x1, y1, x2, y2] format to
    YOLO-style dict rows containing bounding boxes in [x_center, y_center, width, height] relative to the original image
    space.

    Args:
        data (dict): Tracker output with bounding boxes in xyxy format relative to the original image pixel space.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - frame (int): Frame number.
            - id (int): Unique track ID
            - x_center (float): X center of bounding box relative to the original image space.
            - y_center (float): Y center of bounding box relative to the original image space.
            - width (float): Width of bounding box relative to the original image space.
            - height (float): Height of bounding box relative to the original image space.
            - conf (float): Confidence score of the detection.
    """
    yolo_rows = []
    for frame in data["frames"]:
        for fish in frame["fish"]:
            bbox = fish["bbox"]
            left = bbox[0]
            top = bbox[1]
            width = bbox[2] - left
            height = bbox[3] - top

            x_center = left + width / 2
            y_center = top + height / 2

            row = {
                "frame": frame["frame_num"],
                "id": fish["id"],
                "x_center": round(x_center, 3),
                "y_center": round(y_center, 3),
                "width": round(width, 3),
                "height": round(height, 3),
                "conf": round(float(fish["conf"]), 3),
            }
            yolo_rows.append(row)

    return yolo_rows


def yolo_to_mot(bbox: Union[Dict, List], img_width, img_height):
    """Convert a YOLO-formatted bounding box to MOT format.

    YOLO format: [x_center, y_center, width, height]
    MOT format: [bb_left, bb_top, bb_width, bb_height]

    Args:
        bbox (list or dict): Bounding box in YOLO format, either as a list or a dict
                             with keys 'x_center', 'y_center', 'width', 'height'.
        img_width (int or float): Width of the original image.
        img_height (int or float): Height of the original image.

    Returns:
        list or dict: Bounding box in MOT format. If input was a list, returns a list
                      [bb_left, bb_top, bb_width, bb_height]. If input was a dict,
                      adds keys 'bb_left', 'bb_top', 'bb_width', 'bb_height' and returns the dict.
    """
    if isinstance(bbox, list):
        x_center, y_center, width, height = bbox

    if isinstance(bbox, dict):
        x_center, y_center, width, height = (
            bbox["x_center"],
            bbox["y_center"],
            bbox["width"],
            bbox["height"],
        )

    bb_left = round((x_center - width / 2) * img_width, 3)
    bb_top = round((y_center - height / 2) * img_height, 3)
    bb_width = round(width * img_width, 3)
    bb_height = round(height * img_height, 3)

    if isinstance(bbox, list):
        bbox = [bb_left, bb_top, bb_width, bb_height]
    else:
        bbox["bb_left"] = bb_left
        bbox["bb_top"] = bb_top
        bbox["bb_width"] = bb_width
        bbox["bb_height"] = bb_height

    return bbox


def dict_rows_to_mot_format(rows: List[Dict], img_width, img_height) -> List[Dict]:
    """Convert tracking row dictionaries (YOLO format) to MOT formatted list of dictionaries.

    Args:
        rows (list[dict]): List of tracking data rows containing bounding boxes in [x_center, y_center, width,
        height] relative to original image space.
        img_width (int): Original image width.
        img_height (int): Original image height.

    Returns:
        MOT formated dictionary containing bounding boxes in MOT format relative to original image space.
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
    """
    for row in rows:
        row["frame"] = row["frame"] + 1  # MOT is 1-based
        row["id"] = row["id"] + 1  # MOT is 1-based

        row = yolo_to_mot(row, img_width, img_height)

        row["x"] = -1  # Ignore and fill with -1
        row["y"] = -1  # Ignore and fill with -1
        row["z"] = -1  # Ignore and fill with -1

    return rows


def format_single_crossing(
    filename: str,
    metadata: ARISMetadata,
    source_path: str = None,
    track_id: int = None,
    frame: int = None,
    bbox: list = None,
    upstream_direction: str = None,
    crossing_direction: str = None,
    len_outputs: dict = None,
) -> dict:
    """Format a single crossing event for export.

    This uses the same naming conventions in ARISFish Software

    Args:
        filename: Source file name
        metadata: ARIS metadata
        track_id: Fish track ID
        frame: Frame number where crossing occurred. If frame_id_closest_to_mean doesn't exist, default to using
        the frame identified in the Counter.
        bbox: Bounding box coordinates [x_center, y_center, width, height] relative to original image space
        upstream_direction: Upstream direction setting. Value from UpstreamDirectionTypes
        crossing_direction: Which side the fish crossed ("left" or "right")
        len_outputs: Length estimation results

    Returns:
        Formatted crossing dictionary
    """
    if track_id is None:
        return {
            "Source.Name": filename,
            "Source.Path": source_path,
            "Frame#": None,
            "Dir": None,
            "ID": None,
            "bbox": None,
            "metadata": metadata,
            "global_coords_px": None,
            "L(cm)": None,
        }

    len_outputs = len_outputs or {}

    return {
        "Source.Name": filename,
        "Source.Path": source_path,
        "Frame#": len_outputs.get(track_id, {}).get("frame_id_closest_to_mean")
        or frame,
        "Dir": "Up" if upstream_direction == crossing_direction else "Down",
        "ID": track_id,
        "bbox": bbox,
        "metadata": metadata,
        "global_coords_px": len_outputs.get(track_id, {}).get("global_coords_px"),
        "L(cm)": round(
            len_outputs.get(track_id, {}).get("filtered_lengths_cm")
            or FC_SCHEMA["L(cm)"].default,
            2,
        ),
    }

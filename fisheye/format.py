from typing import Union


def tracker_output_to_dict_rows(data: dict):
    """Convert TrackerOutput to YOLO-formatted dict rows.

    Args:
        data (dict): Tracker output with bounding boxes.

    Returns:
        list[dict]: Each row with YOLO-format bbox fields.
    """
    yolo_rows = []
    for frame in data["frames"]:
        for fish in frame["fish"]:
            bbox = fish["bbox"]
            x_center = bbox[0]
            y_center = bbox[1]
            width = bbox[2]
            height = bbox[3]

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


def yolo_to_mot(bbox: Union[dict, list], img_width, img_height):
    """Convert YOLO-formatted bbox to MOT format."""
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


def dict_rows_to_mot_format(rows: list[dict], img_width, img_height) -> list[dict]:
    """Convert tracking row dictionaries (YOLO format) to MOT formatted string.

    Args:
        rows (list[dict]): List of tracking data rows.
        img_width (int): Image width.
        img_height (int): Image height.

    Returns:
        MOT formated dictionary
    """
    for row in rows:
        row["frame"] = row["frame"] + 1  # MOT is 1-based
        row["id"] = row["id"] + 1  # MOT is 1-based

        row = yolo_to_mot(row, img_width, img_height)

        row["x"] = -1  # Ignore and fill with -1
        row["y"] = -1  # Ignore and fill with -1
        row["z"] = -1  # Ignore and fill with -1

        row.pop("x_center")
        row.pop("y_center")
        row.pop("width")
        row.pop("height")

    return rows

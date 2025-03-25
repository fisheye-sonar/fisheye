def tracker_output_to_mot(data: dict):
    """Convert TrackerOutput to MOT format.

    Args:
        data (dict): MOT formatted output.
    """
    mot_rows = []
    for frame in data["frames"]:
        for fish in frame["fish"]:
            bbox = fish["bbox"]
            left = bbox[0]
            top = bbox[1]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            row = {
                "frame": frame["frame_num"] + 1,
                "id": fish["id"] + 1,
                "bb_left": round(left, 3),
                "bb_top": round(top, 3),
                "bb_width": round(w, 3),
                "bb_height": round(h, 3),
                "conf": round(float(fish["conf"]), 3),
            }
            mot_rows.append(row)

    return mot_rows

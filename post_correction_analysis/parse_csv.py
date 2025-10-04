import csv
from col import col


def safe_int(value, default=None):
    try:
        return int(value) if value != "" else default
    except (ValueError, TypeError) as e:
        print(f"{col.yellow}Warning: Could not parse ID: {value} ({e}){col.reset}")
        return default


def safe_float(value, default=None):
    try:
        return float(value) if value != "" else default
    except (ValueError, TypeError) as e:
        print(f"{col.yellow}Warning: Could not parse Theta: {value} ({e}){col.reset}")
        return default


def parse_csv_file(file_path: str):
    """
    Parse a CSV file with the specified format and extract frame_id, direction, R(m), theta, and track_id (if possible).
    """
    data = []
    up_count = 0
    down_count = 0

    try:
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                if all(
                    [
                        "Frame#" in row,
                        "Dir" in row,
                        "Theta" in row,
                        "ID" in row,
                        "R (m)" in row,
                    ]
                ):
                    frame_id = (
                        safe_int(row.get("Frame#")) if row.get("Frame#") else None
                    )
                    direction = row.get("Dir").lower() if row.get("Dir") else None
                    theta = safe_float(row.get("Theta")) if row.get("Theta") else None
                    track_id = (
                        safe_int(row.get("ID"))
                        if row.get("ID", "").isdigit()
                        else frame_id
                    )
                    r_m = safe_float(row.get("R (m)")) if row.get("R (m)") else None

                    # Use ID if available, else fallback to frame_id
                    track_id = (
                        safe_int(row["ID"])
                        if row.get("ID") and row["ID"].isdigit()
                        else frame_id
                    )

                    data.append(
                        {
                            "frame_id": frame_id,
                            "direction": direction,
                            "r_m": r_m,
                            "theta": theta,
                            "track_id": track_id,
                        }
                    )

                    # Count directions
                    if direction == "up":
                        up_count += 1
                    elif direction == "down":
                        down_count += 1
                else:
                    print(
                        f"{col.yellow}Warning: Could not parse row in file {file_path}: {col.reset}{row}"
                    )
                    if "Frame#" not in row:
                        print(
                            f"{col.yellow}    Could not parse as Frame# is missing from the row{col.reset}"
                        )
                    if "Dir" in row:
                        print(
                            f"{col.yellow}    Could not parse as Dir is missing from the row{col.reset}"
                        )
                    if "Theta" not in row:
                        print(
                            f"{col.yellow}    Could not parse as Theta is missing from the row{col.reset}"
                        )
                    if "R (m)" not in row:
                        print(
                            f"{col.yellow}    Could not parse as R (m) is missing from the row{col.reset}"
                        )

    except FileNotFoundError:
        print(f"{col.red}Error: File {file_path} not found{col.reset}")
        return None, 0, 0
    except Exception as e:
        print(f"{col.red}Error reading file {file_path}: {e}{col.reset}")
        return None, 0, 0

    return data, up_count, down_count

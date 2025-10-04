def parse_fc_file(file_path: str):
    """
    Parse a text file with the specified format and extract frame_id, direction, R(m), theta, and (if possible)track_id.
    """
    data = []
    up_count = 0
    down_count = 0

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Skip header lines (lines starting with *** or containing dashes)
        data_lines = []

        for line in lines:
            line = line.strip()

            if (
                line
                and not line.startswith("***")
                and not line.startswith("---")
                and not line.startswith("File")
                and not line == "\n"
            ):
                data_lines.append(line)

        # Parse data lines
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 5:  # Ensure we have enough columns
                try:

                    total_frame = int(parts[1])  # Total column
                    frame_id = int(parts[2])  # Frame# column
                    direction = parts[3].lower()  # Dir column (convert to lowercase)
                    r_m = float(parts[4])  # R (m) column
                    theta = float(parts[5])  # Theta column
                    comments = parts[-1]
                    if "Centerlinecrossingtrackid" in comments:
                        track_id = int(comments.split("Centerlinecrossingtrackid")[1])
                    else:
                        track_id = total_frame

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

                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line}, {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, 0, 0
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, 0, 0

    return data, up_count, down_count

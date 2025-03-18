from pathlib import Path
import glob
import os

import pandas as pd


def load_manual_markings(fp):
    """Load manual markings given to us by fish people. Usually in the form of CSV or xlsx file"""
    # Standardize common column names
    column_mapping = {
        "frame_id": "Frame#",  # Useless operation but want to be consistent with what we are wanting
        "FrameNum": "Frame#",
        "Frame#": "Frame#",
        "Frame": "Frame#",
        "frame": "Frame#",
    }

    if fp.endswith(".csv"):
        df = pd.read_csv(fp)

    elif fp.endswith(".txt"):
        # Match one or more whitespace characters
        df = pd.read_csv(fp, sep="\t", engine="python")

    elif fp.endswith(".xlsx"):
        # CDFW uses excel
        df = pd.read_excel(fp)
        df = df.drop(columns=[df.columns[0]])

        original_col_names = list(df.columns)
        new_col_names = [
            "Hour",
            "39-64cm Up",
            "39-64cm Down",
            "39-64cm Net Movement",
            ">64cm Up",
            ">64cm Down",
            ">64cm Net Movement",
        ]
        updated_col_names = new_col_names + original_col_names[len(new_col_names) :]
        df.columns = updated_col_names

    else:
        raise ValueError("Unsupported file format. Only .csv or .xlsx are allowed.")

    # Pandas will ignore columns that do not exist in the dataframe
    df = df.rename(columns=column_mapping)
    return df


def load_mot_file(
    root_dir: str = "/Users/madison/Documents/Results/2025_van_duzen_inference/mot/",
    deployment_location: str = None,
) -> pd.DataFrame:
    """Load MOT .txt file(s) into Pandas DataFrame.

    Args:
        root_dir (str, optional): Directory where MOT .txt files are located.
        deployment_location (str, optional): Name of river location, e.g. Klamath
    """

    df_list = []
    non_empty_files = None
    text_files = glob.glob(f"{root_dir}/*.txt")

    if text_files:
        non_empty_files = [f for f in text_files if os.path.getsize(f) > 0]

        # Add on the filename without suffix/ext as well as the local path to MOT file
        tmp_df = pd.concat(
            (
                pd.read_csv(f, delimiter=",", header=None).assign(
                    file_stem=Path(f).stem,
                    local_path=f,
                    deployment_location=deployment_location,
                )
                for f in non_empty_files
            ),
            ignore_index=True,
        )
        df_list.append(tmp_df)

    df_detections = pd.concat(df_list, ignore_index=True)
    column_names = [
        "frame_id",
        "fish_id",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "file_stem",
        "local_path",
        "deployment_location",
    ]
    df_detections.columns = column_names

    if non_empty_files:
        print(f"{len(non_empty_files)} empty MOT files found: {non_empty_files}")

    return df_detections

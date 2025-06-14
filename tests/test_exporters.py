import numpy as np
import pandas as pd

from fisheye.configs.datasets import ARISMetadata
from fisheye.export import to_detailed_csv, to_summary_csv, to_txt


def sample_data():
    """Mock output."""
    return [
        [
            {
                "file_name": "2025-06-13_000000.aris",
                "frame_id": 1,
                "fish_id": 1,
                "direction": "left",
                "bbox": [
                    np.float64(0.471),
                    np.float64(0.374),
                    np.float64(0.06),
                    np.float64(0.01),
                ],
                "metadata": ARISMetadata(
                    xdim=773,
                    ydim=1568,
                    image_meter_width=11.368606870117187,
                    image_meter_height=23.060770468749997,
                    pixel_meter_size=0.014707124023437499,
                    x_meter_start=np.float64(-5.681702232746064),
                    x_meter_stop=np.float64(5.686904637371122),
                    y_meter_start=np.float64(23.8129820801439),
                    y_meter_stop=np.float64(0.7522116113939034),
                    sampleperiod=20,
                    soundspeed=1470.71240234375,
                    windowstart=0.7890372276306152,
                    samplesperbeam=1565,
                    BeamCount=48,
                    thesystemtype=2,
                    numframes=500,
                    largelens=0,
                ),
            },
            {
                "file_name": "2025-06-13_000000.aris",
                "frame_id": 2,
                "fish_id": 1,
                "direction": "right",
                "bbox": [
                    np.float64(0.471),
                    np.float64(0.374),
                    np.float64(0.06),
                    np.float64(0.01),
                ],
                "metadata": ARISMetadata(
                    xdim=773,
                    ydim=1568,
                    image_meter_width=11.368606870117187,
                    image_meter_height=23.060770468749997,
                    pixel_meter_size=0.014707124023437499,
                    x_meter_start=np.float64(-5.681702232746064),
                    x_meter_stop=np.float64(5.686904637371122),
                    y_meter_start=np.float64(23.8129820801439),
                    y_meter_stop=np.float64(0.7522116113939034),
                    sampleperiod=20,
                    soundspeed=1470.71240234375,
                    windowstart=0.7890372276306152,
                    samplesperbeam=1565,
                    BeamCount=48,
                    thesystemtype=2,
                    numframes=500,
                    largelens=0,
                ),
            },
        ]
    ]


def test_detailed_csv_creates_file_and_content(tmp_path):
    """Test creating detailed csv per aris file."""
    data = sample_data()
    to_detailed_csv(data, tmp_path, job_id="testjob")

    out_files = list(tmp_path.glob("*.csv"))
    assert len(out_files) == 1

    df = pd.read_csv(out_files[0])
    assert "file_name" in df.columns
    assert df.shape[0] == 2  # 2 rows exported


def test_summary_csv_creates_file_and_content(tmp_path):
    """Test creating summary csv."""
    data = sample_data()
    to_summary_csv(data, tmp_path, job_id="testjob")

    out_files = list(tmp_path.glob("*_summary.csv"))
    assert len(out_files) == 1

    df = pd.read_csv(out_files[0])
    assert "file_name" in df.columns
    assert "net_count" in df.columns
    assert df.shape[0] == 1  # 1 unique file_name

    # Left and right should cancel out → net_count == 0
    assert df["net_count"].iloc[0] == 0


def test_txt_creates_file_and_lines(tmp_path):
    """Test creating ARISFish TXT per aris file."""
    data = sample_data()
    to_txt(data, tmp_path)

    out_files = list(tmp_path.glob("*.txt"))
    assert len(out_files) == 1

    with open(out_files[0], "r") as f:
        lines = f.readlines()

    assert any("*** Manual Marking" in line for line in lines)
    assert any("Frame#" in line for line in lines)
    assert len(lines) > 4


def test_detailed_csv_empty_data(tmp_path):
    """Test passing in empty data for detailed csv."""
    data = []
    to_detailed_csv(data, tmp_path, job_id="testjob")

    out_files = list(tmp_path.glob("*.csv"))
    assert len(out_files) == 0


def test_summary_csv_empty_data(tmp_path):
    """Test passing in empty data for summary csv."""
    data = []
    to_summary_csv(data, tmp_path, job_id="testjob")

    out_files = list(tmp_path.glob("*_summary.csv"))
    assert len(out_files) == 0


def test_txt_empty_data(tmp_path):
    """Test passing in empty data for txt file."""
    data = []
    to_txt(data, tmp_path)

    out_files = list(tmp_path.glob("*.txt"))
    assert len(out_files) == 0

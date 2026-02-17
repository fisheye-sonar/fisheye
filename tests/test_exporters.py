import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from fisheye.configs.datasets import ARISMetadata
from fisheye.export import DetailedCSVExporter, SummaryCSVExporter, FCExporter
from fisheye.export.inference import XMLExporter
from fisheye.utils import convert_pixels_to_coords_meters


def sample_data():
    """Mock output."""
    return [
        [
            {
                "Source.Name": "2025-06-13_000000.aris",
                "Frame#": 1,
                "ID": 1,
                "Dir": "Up",
                "R (m)": 17,
                "Theta": 1.7,
                "global_coords_px": [[466, 588], [502, 604]],
                "L(cm)": 60,
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
                    unwarped_shape=(0, 0),
                    beam_width_data={},
                ),
            },
            {
                "Source.Name": "2025-06-13_000000.aris",
                "Frame#": 2,
                "ID": 2,
                "Dir": "Down",
                "R (m)": 17,
                "Theta": 1.7,
                "global_coords_px": [[600, 700], [703, 826]],
                "L(cm)": 80,
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
                    unwarped_shape=(0, 0),
                    beam_width_data={},
                ),
            },
        ]
    ]


def test_detailed_csv_creates_file_and_content(tmp_path):
    """Test creating detailed csv per aris file."""
    data = sample_data()
    exporter = DetailedCSVExporter(output_dir=str(tmp_path), job_id="testjob")
    exporter.export(data)

    out_files = list(tmp_path.glob("*.csv"))
    assert len(out_files) == 1

    df = pd.read_csv(out_files[0])
    assert len(df.columns) == 29
    assert len(df) == 2


def test_detailed_csv_empty_data(tmp_path):
    """Test passing in empty data for detailed csv."""
    data = []
    exporter = DetailedCSVExporter(output_dir=str(tmp_path), job_id="testjob")
    exporter.export(data)

    out_files = list(tmp_path.glob("*.csv"))
    assert len(out_files) == 0


def test_summary_csv_creates_file_and_content(tmp_path):
    """Test creating summary csv."""
    data = sample_data()
    exporter = SummaryCSVExporter(output_dir=str(tmp_path), job_id="testjob")
    exporter.export(data)

    out_files = list(tmp_path.glob("*_summary.csv"))
    assert len(out_files) == 1

    df = pd.read_csv(out_files[0])
    assert "Source.Name" in df.columns
    assert "net_count" in df.columns
    assert len(df) == 1

    # Left vs right cancel → 0
    assert df["net_count"].iloc[0] == 0


def test_summary_csv_empty_data(tmp_path):
    """Test passing in empty data for summary csv."""
    data = []
    exporter = SummaryCSVExporter(output_dir=str(tmp_path), job_id="testjob")
    exporter.export(data)

    out_files = list(tmp_path.glob("*_summary.csv"))
    assert len(out_files) == 0


def test_txt_creates_file_and_lines(tmp_path):
    """Test creating ARISFish TXT per aris file."""
    data = sample_data()
    exporter = FCExporter(output_dir=str(tmp_path), job_id="testjob")
    exporter.export(data)

    out_files = list(tmp_path.glob("FCe_*.txt"))
    assert len(out_files) == 1

    with open(out_files[0], "r") as f:
        lines = f.readlines()

    assert any("*** Manual Marking" in line for line in lines)
    assert any("Frame#" in line for line in lines)
    assert len(lines) > 4


def test_txt_empty_data(tmp_path):
    """Test passing in empty data for txt file."""
    data = []
    exporter = FCExporter(output_dir=str(tmp_path), job_id="testjob")
    exporter.export(data)

    out_files = list(tmp_path.glob("FCe_*.txt"))
    assert len(out_files) == 0


def test_xml_empty_data(tmp_path):
    """Test exporting empty data still creates XML file."""
    data = [[{"Source.Name": "2025-06-13_000000.aris", "global_coords_px": None}]]
    exporter = XMLExporter(output_dir=str(tmp_path), job_id="testjob")
    exporter.export(data=data)

    xml_files = list(tmp_path.glob("*.xml"))
    assert len(xml_files) == 1
    tree = ET.parse(xml_files[0])
    root = tree.getroot()
    assert root.tag == "MarkedFishMeasurements"
    assert len(root) == 1


def test_xml_with_sample_data(tmp_path):
    """Test exporting sample data to XML file."""
    data = sample_data()
    world_coords1 = convert_pixels_to_coords_meters(
        data[0][0]["global_coords_px"], data[0][0]["metadata"]
    )
    world_coords2 = convert_pixels_to_coords_meters(
        data[0][1]["global_coords_px"], data[0][1]["metadata"]
    )

    exporter = XMLExporter(
        output_dir=str(tmp_path), job_id="testjob", upstream_direction="left"
    )
    exporter.export(data=data)

    xml_files = list(tmp_path.glob("*.xml"))
    assert len(xml_files) == 1

    tree = ET.parse(xml_files[0])
    root = tree.getroot()

    measurements = root.findall("MarkedFishMeasurement")
    assert len(measurements) == 2

    # First measurement
    m1 = measurements[0]
    assert m1.attrib["FishID"] == "1"
    assert m1.attrib["FrameIndex"] == "1"

    nodes1 = m1.findall("FishMeasureNode")
    assert len(nodes1) == 2
    assert nodes1[0].attrib["Length"] == "0"
    assert nodes1[0].attrib["WorldPointX"] == str(world_coords1[0][0])
    assert nodes1[0].attrib["WorldPointY"] == str(world_coords1[0][1])
    assert nodes1[1].attrib["Length"] == "60"
    assert nodes1[1].attrib["WorldPointX"] == str(world_coords1[1][0])
    assert nodes1[1].attrib["WorldPointY"] == str(world_coords1[1][1])

    # Second measurement
    m2 = measurements[1]
    assert m2.attrib["FishID"] == "2"
    assert m2.attrib["FrameIndex"] == "2"

    nodes2 = m2.findall("FishMeasureNode")
    assert len(nodes2) == 2
    assert nodes2[0].attrib["Length"] == "0"  # [[600, 700], [703, 826]],
    assert nodes2[0].attrib["WorldPointX"] == str(world_coords2[1][0])
    assert nodes2[0].attrib["WorldPointY"] == str(world_coords2[1][1])
    assert nodes2[1].attrib["Length"] == "80"
    assert nodes2[1].attrib["WorldPointX"] == str(world_coords2[0][0])
    assert nodes2[1].attrib["WorldPointY"] == str(world_coords2[0][1])

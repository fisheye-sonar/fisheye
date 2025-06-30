import argparse
import time
from pathlib import Path
from typing import List, Union

import structlog

from fisheye.common.generic import load_model_config
from fisheye.common.logging import setup_logging
from fisheye.common.system import check_disk_space, generate_job_id
from fisheye.configs import YOLOv5ModelConfig, ObjectDetectionConfig
from fisheye.enums import ExportType
from fisheye.export import save_to_disk
from fisheye.pipelines.pipeline import DetectTrackCountPipeline
from fisheye.version import __app_version__, __detector_version__

job_id = generate_job_id()
setup_logging(file_logging=True, job_id=job_id)
logger = structlog.get_logger().bind(
    job_id=job_id, app_version=__app_version__, detector_version=__detector_version__
)


def main(
    path: Union[List[str], str],
    export_options: List[ExportType],
    output_dir: str,
    map_input_dir_structure_to_output: bool = False,
):
    check_disk_space(path=output_dir)  # Make sure there's enough space to store results
    project_root = Path(__file__).resolve().parents[1]
    relative_model_path = load_model_config()["detector"]["path"]
    model_path = str((project_root / relative_model_path).resolve())

    model_cfg = YOLOv5ModelConfig(weights=model_path)
    detection_cfg = ObjectDetectionConfig(model=model_cfg)

    start_time = time.time()
    logger.info("inference_started", start_time=start_time)

    results = DetectTrackCountPipeline(detector_cfg=detection_cfg).run(
        path,
        output_dir,
        export_options,
        job_id,
        map_input_dir_structure_to_output=map_input_dir_structure_to_output,
    )

    if ExportType.SUMMARY_CSV in export_options:
        save_to_disk(
            results, output_dir, export_types=ExportType.SUMMARY_CSV, job_id=job_id
        )

    end_time = time.time()
    logger.info(
        "inference_completed",
        inference_duration_sec=end_time - start_time,
        num_files_processed=len(results),
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="Path to directory of ARIS/DIDSON files.",
    )

    parser.add_argument(
        "--export_options",
        required=False,
        type=str,
        default="summary_csv,detailed_csv,txt",
        help="Comma-separated list of export types.",
    )

    parser.add_argument(
        "--map_input_dir_structure_to_output",
        action=argparse.BooleanOptionalAction,
        help="Map input directory structure to output directory structure.",
    )

    parser.add_argument(
        "--output_dir",
        required=False,
        type=str,
        help="Directory to save results. If results are saved in the same location as ARIS/DIDSON files, they can be "
        "ingested by ARISFish Software from Sound Metrics.",
    )
    args = parser.parse_args()

    parts = [v.strip().upper() for v in args.export_options.split(",")]
    export_types = []
    for p in parts:
        try:
            export_types.append(ExportType[p])
        except KeyError as e:
            raise argparse.ArgumentTypeError(f"Invalid export type: {e.args[0]}")

    results = main(
        args.path,
        export_types,
        args.output_dir,
        map_input_dir_structure_to_output=args.map_input_dir_structure_to_output,
    )

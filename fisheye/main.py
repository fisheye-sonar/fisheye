import argparse
import logging
import time
from typing import List

from fisheye.common.logging import setup_logging
from fisheye.common.system import check_disk_space
from fisheye.configs import YOLOv5ModelConfig, ObjectDetectionConfig
from fisheye.enums import ExportType
from fisheye.export import save_to_disk
from fisheye.pipelines.pipeline import DetectTrackCountPipeline

setup_logging(modules=["dataloaders", "pipelines", "track", "count", "export"])

logger = logging.getLogger(__name__)


def main(
    path: List[str] | str, weights, export_options: List[ExportType], output_dir: str
):
    check_disk_space(path="/")  # Make sure there's enough space to store results

    model_cfg = YOLOv5ModelConfig(weights=weights)
    detection_cfg = ObjectDetectionConfig(model=model_cfg)
    logger.info("Pipeline started 🚀")
    # TODO (MVH) - this may take up too much memory holding all of the results, probably need to dump earlier
    results = DetectTrackCountPipeline(detection_cfg).run(
        path, output_dir, export_options
    )

    if ExportType.SUMMARY_CSV in export_options:
        save_to_disk(results, output_dir, export_types=ExportType.SUMMARY_CSV)

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
        "--weights", required=True, type=str, help="Path to model weights."
    )

    parser.add_argument(
        "--export_options",
        required=False,
        type=str,
        default="summary_csv,detailed_csv,txt",
        help="Comma-separated list of export types.",
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

    start = time.time()
    results = main(args.path, args.weights, export_types, args.output_dir)
    end = time.time()

    logger.info(f"Total inference time: {end - start:.2f} seconds")

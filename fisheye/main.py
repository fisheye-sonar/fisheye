import argparse
import logging
import time
from typing import List

from fisheye.configs import YOLOv5ModelConfig, ObjectDetectionConfig
from fisheye.export import get_exporter
from fisheye.logging import setup_logging
from fisheye.pipelines.pipeline import DetectTrackCountPipeline


setup_logging(modules=["dataloaders", "pipelines", "track", "count", "export"])

logger = logging.getLogger(__name__)


def main(path: List[str] | str, weights, export_format: str, output_dir: str):
    model_cfg = YOLOv5ModelConfig(weights=weights)
    detection_cfg = ObjectDetectionConfig(model=model_cfg)
    logger.info("Pipeline started 🚀")
    results = DetectTrackCountPipeline(detection_cfg).run(path)

    if export_format:
        export_function = get_exporter(export_format)

        if export_function:
            export_function(results, output_dir)

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
        "--export_format",
        required=False,
        type=str,
        default=None,
        choices=["csv", None],
        help="Export results to 'csv' or 'text' format. Leave empty for no export.",
    )

    parser.add_argument(
        "--output_dir", required=False, type=str, help="Path to save results."
    )
    args = parser.parse_args()
    start = time.time()
    results = main(args.path, args.weights, args.export_format, args.output_dir)
    end = time.time()

    logger.info(f"Total inference time: {end - start:.2f} seconds")

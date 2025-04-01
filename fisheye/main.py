import argparse
import time
from typing import List

from fisheye.configs import YOLOv5ModelConfig, ObjectDetectionConfig
from fisheye.export import to_csv
from fisheye.pipelines.pipeline import DetectTrackCountPipeline


def main(path: List[str] | str, weights):
    model_cfg = YOLOv5ModelConfig(weights=weights)
    detection_cfg = ObjectDetectionConfig(model=model_cfg)

    results = DetectTrackCountPipeline(detection_cfg).run(path)

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
        "--export",
        required=False,
        type=str,
        choices=["csv", "text", None],
        help="Export results to 'csv' or 'text' format. Leave empty for no export.",
    )

    parser.add_argument(
        "--out_path", required=False, type=str, help="Path to save results."
    )
    args = parser.parse_args()
    start = time.time()
    output = main(args.path, args.weights)
    end = time.time()

    print(f"Total inference time: {end - start:.2f} seconds")

    if args.export == "csv":
        to_csv(output, args.out_path)

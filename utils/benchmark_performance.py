import argparse
import csv
import datetime
import itertools
import subprocess
import time

from memory_profiler import memory_usage

from fisheye.common.generic import cleanup, set_seed
from fisheye.configs import YOLOv5ModelConfig, ObjectDetectionConfig, YOLODatasetConfig
from fisheye.pipelines import ObjectDetectionPipeline


DEFAULT_BATCH_SIZES = [2, 4, 8, 16, 32]
DEFAULT_MAX_WORKERS = [1, 2, 4, 8]
DEFAULT_NUM_WORKERS = 0


def run_detector_pipeline(dataset_config, detection_cfg):
    """Runs dataloader and detector pipeline."""
    detector = ObjectDetectionPipeline(detection_cfg, dataset_config)
    detections = detector()

    del detector
    del detection_cfg

    return detections


def run_pipeline_with_memory_tracking(dataset_config, detection_cfg):
    def wrapped():
        start = time.perf_counter()
        result = run_detector_pipeline(dataset_config, detection_cfg)
        end = time.perf_counter()
        duration = end - start

        return result, duration

    try:
        mem_usage, (result, duration) = memory_usage(
            wrapped,
            retval=True,
            interval=0.5,
            timeout=None,
            include_children=True,
            multiprocess=True,
            stream=None,
        )

        mem_increment = max(mem_usage) - min(mem_usage)
        print(f"[MEMORY] Increment: {mem_increment:.2f} MiB", flush=True)
        print(f"[TIMER] Pipeline took {duration:.4f} seconds", flush=True)

        return duration, mem_increment

    except Exception as e:
        print(f"[ERROR] memory_usage failed: {e}", flush=True)
        raise


def benchmark_batch_sizes(weights, fp):
    batch_sizes = DEFAULT_BATCH_SIZES
    nw = DEFAULT_NUM_WORKERS
    max_workers = DEFAULT_MAX_WORKERS

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    with open(
        f"../fisheye/logs/{date}_benchmark_performance.csv",
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Batch Size",
                "Max Workers",
                "Time Taken (seconds)",
                "Memory Increment (MiB)",
                "File",
            ]
        )

        for batch_size, mw in itertools.product(batch_sizes, max_workers):
            print(f"\n>>> Current batch size: {batch_size} and number of workers: {mw}")
            result = subprocess.run(
                [
                    "python",
                    "profiler.py",
                    "--mode",
                    "run",
                    "--batch_size",
                    str(batch_size),
                    "--num_workers",
                    str(nw),
                    "--max_workers",
                    str(mw),
                    "--weights",
                    weights,
                    "--filepath",
                    fp,
                ],
                capture_output=True,
                text=True,
            )
            print(result.stdout)

            time_line = next(
                (line for line in result.stdout.splitlines() if "[TIMER]" in line), None
            )
            mem_line = next(
                (line for line in result.stdout.splitlines() if "[MEMORY]" in line),
                None,
            )

            if time_line and mem_line:
                time_taken = float(time_line.split("took")[1].split("seconds")[0])
                mem_increment = float(mem_line.split("Increment:")[1].split("MiB")[0])
                writer.writerow([batch_size, mw, time_taken, mem_increment, fp])

            cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="run", help="Mode: run or benchmark"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--weights", type=str, help="Model weights path")
    parser.add_argument("--filepath", type=str, help="Path to input .aris file")
    parser.add_argument("--num_workers", type=int, help="Number of workers")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Max workers for threading (e.g., inference threads)",
    )

    args = parser.parse_args()

    set_seed(42)
    if args.mode == "benchmark":
        benchmark_batch_sizes(args.weights, args.filepath)

    else:
        dataset_cfg = YOLODatasetConfig(
            filepath=args.filepath,
            batch_size=args.batch_size,
            workers=args.num_workers,
            max_workers=args.max_workers,
        )

        model_cfg = YOLOv5ModelConfig(weights=args.weights)
        detection_cfg = ObjectDetectionConfig(
            model=model_cfg, max_workers=args.max_workers
        )

        time_taken, mem_increment = run_pipeline_with_memory_tracking(
            dataset_cfg, detection_cfg
        )

        cleanup()

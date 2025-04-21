import argparse
import csv
import gc
import itertools
import random
import subprocess
import time

import numpy as np
import torch
from memory_profiler import memory_usage

from fisheye.configs import YOLOv5ModelConfig, ObjectDetectionConfig, YOLODatasetConfig
from fisheye.pipelines import ObjectDetectionPipeline


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup():
    """Clear up memory between runs."""
    torch.cuda.empty_cache()
    gc.collect()


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMER] {func.__name__} took {end - start:.4f} seconds")
        return result, end - start

    return wrapper


# @timed
def run_detector_pipeline(weights, fp, dataset_config):
    """Runs dataloader and detector pipeline."""

    model_cfg = YOLOv5ModelConfig(weights=weights)
    detection_cfg = ObjectDetectionConfig(model=model_cfg)

    detector = ObjectDetectionPipeline(detection_cfg, dataset_config)
    detections = detector()

    del detector  # Free up memory
    del model_cfg
    del detection_cfg

    return detections


def run_pipeline_with_memory_tracking(weights, fp, dataset_config):
    def wrapped():
        # return run_detector_pipeline(weights, fp, dataset_config)
        start = time.perf_counter()
        result = run_detector_pipeline(weights, fp, dataset_config)
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
        print(max(mem_usage), min(mem_usage))
        mem_increment = max(mem_usage) - min(mem_usage)
        print(f"[MEMORY] Increment: {mem_increment:.2f} MiB", flush=True)
        print(f"[TIMER] Pipeline took {duration:.4f} seconds", flush=True)
        return duration, mem_increment

    except Exception as e:
        print(f"[ERROR] memory_usage failed: {e}", flush=True)
        raise


def benchmark_batch_sizes(weights, fp):
    batch_sizes = [2, 4, 8, 16, 32]
    # num_workers = [0, 2, 4, 8]
    nw = 0
    max_workers = [1, 2, 4, 8]

    with open(
        "logs/2025-04-19_benchmark_object_detection_pipeline_multithreading.csv",
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
        # run_detector_pipeline(args.weights, args.filepath, dataset_cfg)
        time_taken, mem_increment = run_pipeline_with_memory_tracking(
            args.weights, args.filepath, dataset_cfg
        )

        cleanup()

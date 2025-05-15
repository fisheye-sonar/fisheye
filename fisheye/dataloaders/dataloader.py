import os
import warnings
from typing import Union

import structlog
import torch

from fisheye.configs import BaseDatasetConfig, YOLODatasetConfig
from fisheye.dataloaders import ARISBatchedDataset, YOLOARISBatchedDataset
from fisheye.dataloaders.samplers import OnePerBatchSampler
from fisheye.common.collate import yolo_collate_fn
from fisheye.common import torch_distributed_zero_first

logger = structlog.get_logger()


def create_dataloader(config: Union[BaseDatasetConfig, YOLODatasetConfig]):
    """
    Get a PyTorch Dataset and DataLoader for ARIS or YOLO dataset files, depending on the config type.
    """
    dataset_class = None
    collate_fn = None

    # Check if config is for ARIS or YOLO dataset and choose corresponding dataset class
    if isinstance(config, YOLODatasetConfig):
        dataset_class = YOLOARISBatchedDataset
        collate_fn = yolo_collate_fn

    elif isinstance(config, BaseDatasetConfig):
        dataset_class = ARISBatchedDataset
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")

    # Make sure only the first process in DDP processes the dataset first, and the following others can use the cache
    # this is a no-op for a single-gpu machine
    with torch_distributed_zero_first(config.rank):
        dataset = dataset_class(config)

    if len(dataset) == 0:
        warnings.warn(
            "Warning: Dataset contains no valid frames or has incorrect start and end frame indexes, "
            "preventing frame extraction."
        )
        return None, None

    batch_size = min(config.batch_size, len(dataset))
    nw = min(
        [
            os.cpu_count() // config.world_size,
            batch_size if batch_size > 1 else 0,
            config.workers,
        ]
    )  # number of workers

    logger.info(
        "Initialized dataloader",
        config_type=type(config).__name__,
        dataset_class=dataset_class.__name__,
        dataset_size=len(dataset),
        batch_size=batch_size,
        num_workers=nw,
    )

    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset,
        batch_size=None,
        sampler=OnePerBatchSampler(data_source=dataset, batch_size=batch_size),
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataloader, dataset

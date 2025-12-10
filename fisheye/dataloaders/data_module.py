import os

import pytorch_lightning as pl
import structlog
from torch.utils.data import DataLoader

from fisheye.common import (
    torch_distributed_zero_first,
    yolo_collate_fn,
    yolo_collate_fn_already_batched,
)
from fisheye.configs.datasets import BaseDatasetConfig
from fisheye.dataloaders.samplers import OnePerBatchSampler

logger = structlog.get_logger()


class ARISDataModule(pl.LightningDataModule):
    """ARISDataModule

    A PyTorch Lightning DataModule for ARIS data.
    """

    def __init__(self, dataset_cls, dataset_config: BaseDatasetConfig):
        """
        Args:
            dataset_cls (Dataset): The dataset class to be used (e.g., ARISBatchedDataset, YOLOBatchedDataset)
            dataset_kwargs (dict): Arguments to initialize the dataset class.
            batch_size (int): Batch size for the dataloader.
            num_workers (int): Number of workers for data loading.
            world_size (int): Number of distributed processes.
            rank (int): Rank of the current process in distributed training.
        """
        super().__init__()
        self.dataset_cls = dataset_cls
        self.dataset_config = dataset_config
        self.batch_size = dataset_config.batch_size
        self.num_workers = dataset_config.workers
        self.world_size = dataset_config.world_size
        self.rank = dataset_config.rank
        self.dataset = None
        self.dataloader = None
        self.preprocess_batchwise = dataset_config.preprocess_batchwise
        self.collate_fn = (
            yolo_collate_fn_already_batched
            if self.preprocess_batchwise
            else yolo_collate_fn
        )

    def setup(self, stage=None):
        """Setup dataset. Called once before training/validation starts."""
        # Ensure only the first DDP process initializes the dataset
        with torch_distributed_zero_first(self.rank):
            self.dataset = self.dataset_cls(self.dataset_config)

        self.batch_size = min(self.batch_size, len(self.dataset))
        self.num_workers = min(
            [
                os.cpu_count() // self.world_size,
                self.batch_size if self.batch_size > 1 else 0,
                self.num_workers,
            ]
        )

        logger.info(
            "Initialized dataloader",
            dataset_size=len(self.dataset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def get_dataloader(self):
        """Returns a DataLoader with the correct collate function."""
        collate_fn = (
            self.collate_fn
            if self.dataset_cls.__name__ == "YOLOARISBatchedDataset"
            else None
        )

        return DataLoader(
            self.dataset,
            batch_size=None,
            sampler=OnePerBatchSampler(
                data_source=self.dataset, batch_size=self.batch_size
            ),
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        """Returns the training dataloader."""
        return self.get_dataloader()

    def val_dataloader(self):
        return self.get_dataloader()

    def test_dataloader(self):
        return self.get_dataloader()

import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from fisheye.dataloaders.samplers import OnePerBatchSampler
from fisheye.utils import torch_distributed_zero_first, yolo_collate_fn

BASE = Path(__file__).parent.parent
BEAM_WIDTH_DIR = (BASE / "beam_widths").resolve()


class ARISDataModule(pl.LightningDataModule):
    """ARISDataModule

    A PyTorch Lightning DataModule for ARIS data.
    """

    def __init__(
        self,
        dataset_cls,
        dataset_kwargs,
        batch_size=32,
        num_workers=0,
        world_size=1,
        rank=-1,
        disable_output=False,
    ):
        """
        Args:
            dataset_cls (Dataset): The dataset class to be used (e.g., ARISBatchedDataset, YOLOBatchedDataset)
            dataset_kwargs (dict): Arguments to initialize the dataset class.
            batch_size (int): Batch size for the dataloader.
            num_workers (int): Number of workers for data loading.
            world_size (int): Number of distributed processes.
            rank (int): Rank of the current process in distributed training.
            disable_output (bool): Whether to disable console output.
        """
        super().__init__()
        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.world_size = world_size
        self.rank = rank
        self.disable_output = disable_output
        self.dataset = None
        self.dataloader = None

    def setup(self, stage=None):
        """
        Setup dataset. Called once before training/validation starts.
        """
        # Ensure only the first DDP process initializes the dataset
        with torch_distributed_zero_first(self.rank):
            self.dataset = self.dataset_cls(**self.dataset_kwargs)

        self.batch_size = min(self.batch_size, len(self.dataset))
        self.num_workers = min(
            [
                os.cpu_count() // self.world_size,
                self.batch_size if self.batch_size > 1 else 0,
                self.num_workers,
            ]
        )

        if not self.disable_output:
            print(f"Dataset size: {len(self.dataset)}")
            print(f"Num workers: {self.num_workers}")

    def get_dataloader(self):
        """Returns a DataLoader with the correct collate function."""
        collate_fn = (
            yolo_collate_fn
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

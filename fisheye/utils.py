from contextlib import contextmanager

import torch


@contextmanager
def torch_distributed_zero_first(rank):
    if rank != -1:
        torch.distributed.barrier()
    yield
    if rank != -1:
        torch.distributed.barrier()


def yolo_collate_fn(batch):
    """See ScaledYOLOv4.utils.datasets.collate_fn"""

    img, label, shapes = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0), shapes

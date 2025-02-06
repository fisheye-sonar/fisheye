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
    if not len(batch):
        print("help!")
        print(batch)

    img, label, shapes = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0), shapes


def select_device(device='cpu', batch_size=32):
    """Select the appropriate device (CPU or CUDA).

    Args:
        device: device string (e.g., 'cpu' or 'cuda')
        batch_size: batch size (needed for some specific configurations)

    Returns:
        device: selected torch device
    """
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

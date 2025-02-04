from contextlib import contextmanager

import torch


@contextmanager
def torch_distributed_zero_first(rank):
    if rank != -1:
        torch.distributed.barrier()
    yield
    if rank != -1:
        torch.distributed.barrier()

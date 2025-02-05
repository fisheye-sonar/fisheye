import numpy as np
import torch


class OnePerBatchSampler(torch.utils.data.Sampler):
    """Yields the first index of each batch, given a batch size.
    In other words, returns multiples of self.batch_size up to the size of the Dataset.
    This is a workaround for Pytorch's standard batch creation that allows us to manually
    select contiguous segments of an ARIS clip for each batch.
    """
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        idxs = [i*self.batch_size for i in range(len(self))]
        return iter(idxs)

    def __len__(self):
        return int(np.ceil(len(self.data_source) / self.batch_size))
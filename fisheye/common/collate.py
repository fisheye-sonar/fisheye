import torch


def yolo_collate_fn(batch):
    """See ScaledYOLOv4.utils.datasets.collate_fn"""

    img, label, shapes = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0), shapes


def yolo_collate_fn_already_batched(batch):
    return batch

import torch


def yolo_collate_fn(batch):
    """See ScaledYOLOv4.utils.datasets.collate_fn"""

    img, label, shapes, img_original = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return (
        torch.stack(img, 0),
        torch.cat(label, 0),
        shapes,
        torch.stack(img_original, 0),
    )

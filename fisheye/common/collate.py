import torch


def yolo_collate_fn(batch):
    """See ScaledYOLOv4.utils.datasets.collate_fn"""

    sample_size = len(batch[0])
    if sample_size not in (4, 5) or any(len(sample) != sample_size for sample in batch):
        raise ValueError("YOLO samples must consistently contain 4 or 5 fields")
    img, label, shapes, img_original = zip(*(sample[:4] for sample in batch))
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()

    output = (
        torch.stack(img, 0),
        torch.cat(label, 0),
        shapes,
        (
            torch.stack(img_original, 0)
            if not all(x is None for x in img_original)
            else None
        ),
    )
    if sample_size == 5:
        captures = [sample[4] for sample in batch]
        preprocessing = captures[0]["preprocessing"]
        if any(capture["preprocessing"] != preprocessing for capture in captures):
            raise ValueError("Endpoint preprocessing differs within one batch")
        output += (
            {
                "frame_ids": tuple(
                    frame_id
                    for capture in captures
                    for frame_id in capture["frame_ids"]
                ),
                "images": tuple(
                    image for capture in captures for image in capture["images"]
                ),
                "preprocessing": preprocessing,
            },
        )
    return output

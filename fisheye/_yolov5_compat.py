"""Keep legacy YOLOv5 checkpoint loading independent of Ultralytics patches."""

import torch


class _CheckpointTorch:
    """Override loading only inside YOLOv5's legacy checkpoint module."""

    def load(self, *args, **kwargs):
        # Trusted YOLOv5 detector checkpoints contain pickled nn.Module objects.
        # Preserve explicit weights_only=True requests and torch.load elsewhere.
        kwargs.setdefault("weights_only", False)
        return torch.load(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(torch, name)


def prepare_yolov5_loading():
    import yolov5

    if not callable(getattr(yolov5, "load", None)):
        location = getattr(yolov5, "__file__", None) or getattr(
            yolov5, "__path__", None
        )
        raise ImportError(
            f"Expected the installed yolov5 package with yolov5.load; imported {location}. "
            "Check for a local yolov5 directory shadowing the installed package."
        )

    from yolov5.models import experimental

    if experimental.torch is torch:
        experimental.torch = _CheckpointTorch()

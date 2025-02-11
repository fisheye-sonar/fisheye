import torch
from ultralytics.nn import torch_safe_load

from fisheye.config import YOLODatasetConfig
from fisheye.models.base import BaseModel


class Ensemble(torch.nn.ModuleList):
    """Ensemble of models.

    Class from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L679
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Method generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


class YOLOv5(BaseModel):
    """
    YOLOv5 object detection model class for inference.
    """

    def __init__(self, model_path, device, config: YOLODatasetConfig) -> None:
        """Initializes the YOLOv5 model by loading weights and setting the device.

        Args:
            model_path (Union[str, Path]): Local path to the YOLO model weights.
            device (torch.device): The device (CPU or GPU) to run inference on.
        """
        super().__init__(model_path, device, config)

    def _load_weights(self, weights, inplace=True, fuse=True):
        """Loads the model weights. Modified version from Ultralytics."""
        ensemble = Ensemble()

        for w in weights if isinstance(weights, list) else [weights]:
            ckpt, w = torch_safe_load(w)  # load ckpt
            model = (
                (ckpt.get("ema") or ckpt["model"]).to(self.device).float()
            )  # FP32 model

            model.pt_path = w  # attach *.pt file path to model

            if not hasattr(model, "stride"):
                model.stride = torch.tensor([32.0])

            if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
                ckpt.names = dict(enumerate(ckpt.names))

            ensemble.append(
                model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()
            )

        # Module updates
        for m in ensemble.modules():
            if hasattr(m, "inplace"):
                m.inplace = inplace
                if not isinstance(m.anchor_grid, list):
                    delattr(m, "anchor_grid")
                    setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
            elif isinstance(m, torch.nn.Upsample) and not hasattr(
                m, "recompute_scale_factor"
            ):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        if len(ensemble) == 1:
            return ensemble[-1]

        for k in "names", "nc", "yaml":
            setattr(ensemble, k, getattr(ensemble[0], k))
        ensemble.stride = ensemble[
            int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))
        ].stride
        assert all(
            ensemble[0].nc == m.nc for m in ensemble
        ), f"Models differ in class counts {[m.nc for m in ensemble]}"

        return ensemble

"""Inference-only port of length_annotator TargetHeatmapModel; see PROVENANCE.md."""

import torch
from torch import nn
import torch.nn.functional as F


class TargetHeatmapModel(nn.Module):
    def __init__(self, pose, cfg):
        super().__init__()
        self.yaml = pose.yaml
        self.features = nn.ModuleList(list(pose.model.children())[:-1])
        self.saved_indices = set(pose.save)
        self.feature_indices = pose.model[-1].f
        channels = [branch[0].conv.in_channels for branch in pose.model[-1].cv4]
        width = cfg["experiment"].get("head_channels", 64)
        self.projections = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.head = nn.Sequential(
            nn.Conv2d(width * len(channels), width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, 2, 1),
        )

    def forward(self, x):
        cache = []
        for layer in self.features:
            if layer.f != -1:
                x = (
                    cache[layer.f]
                    if isinstance(layer.f, int)
                    else [x if j == -1 else cache[j] for j in layer.f]
                )
            x = layer(x)
            cache.append(x if layer.i in self.saved_indices else None)
        maps = [cache[i] for i in self.feature_indices]
        size = (maps[0].shape[-2] * 2, maps[0].shape[-1] * 2)
        fused = [
            F.interpolate(project(f), size=size, mode="bilinear", align_corners=False)
            for project, f in zip(self.projections, maps)
        ]
        return self.head(torch.cat(fused, 1))

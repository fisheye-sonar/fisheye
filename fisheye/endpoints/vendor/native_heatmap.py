import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_input_to_min_size(x, *, min_height=8, min_width=8):
    height, width = x.shape[-2:]
    pad_h = max(0, int(min_height) - int(height))
    pad_w = max(0, int(min_width) - int(width))
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, double_conv=True):
        super().__init__()
        if double_conv:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, padding=p),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, k, padding=p),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, padding=p),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_double_conv=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = Block(in_ch + skip_ch, out_ch, double_conv=use_double_conv)

    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(
                skip,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HeatmapCNN(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        height, width = x.shape[-2:]
        x = _pad_input_to_min_size(x)

        x = self.down(x)
        x = self.middle(x)
        x = self.up(x)
        x = self.head(x)

        if x.shape[-2:] != (height, width):
            x = F.interpolate(
                x,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        return x


class UNetHeatmap(nn.Module):
    def __init__(
        self,
        in_ch=1,
        out_ch=2,
        base=64,
        use_double_conv=True,
    ):
        super().__init__()
        self.inc = Block(in_ch, base, double_conv=use_double_conv)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            Block(base, base * 2, double_conv=use_double_conv),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            Block(base * 2, base * 4, double_conv=use_double_conv),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            Block(base * 4, base * 4, double_conv=use_double_conv),
        )
        self.mid = Block(base * 4, base * 4, double_conv=use_double_conv)
        self.up3 = Up(
            in_ch=base * 4,
            skip_ch=base * 4,
            out_ch=base * 2,
            use_double_conv=use_double_conv,
        )
        self.up2 = Up(
            in_ch=base * 2,
            skip_ch=base * 2,
            out_ch=base,
            use_double_conv=use_double_conv,
        )
        self.up1 = Up(
            in_ch=base,
            skip_ch=base,
            out_ch=base,
            use_double_conv=use_double_conv,
        )
        self.head = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        height, width = x.shape[-2:]
        x = _pad_input_to_min_size(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xm = self.mid(x4)

        x = self.up3(xm, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.head(x)

        if x.shape[-2:] != (height, width):
            x = F.interpolate(
                x,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # Accept (B, H, W) or (B, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        H, W = x.shape[-2:]

        x = self.down(x)
        x = self.middle(x)
        x = self.up(x)
        x = self.head(x)

        # Guarantee exact size match (handles odd sizes / rounding)
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x


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
    """Upsample + concat skip + DoubleConv"""

    def __init__(self, in_ch, skip_ch, out_ch, use_double_conv=True):
        """
        in_ch:  channels coming from the previous layer (before concat)
        skip_ch: channels from the skip connection
        out_ch: output channels after the double conv
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = Block(in_ch + skip_ch, out_ch, double_conv=use_double_conv)

    def forward(self, x, skip):
        x = self.up(x)
        # In case sizes differ due to odd pooling, resize the skip to match x
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(
                skip, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetHeatmap(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base=64, use_double_conv=True):
        super().__init__()

        # Encoder
        self.inc = Block(in_ch, base, double_conv=use_double_conv)  # -> 64
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), Block(base, base * 2, double_conv=use_double_conv)
        )  # -> 128
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), Block(base * 2, base * 4, double_conv=use_double_conv)
        )  # -> 256

        # (Optional) extra depth to mirror your original 3 pools
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), Block(base * 4, base * 4, double_conv=use_double_conv)
        )  # keep 256 to match your middle

        # Bottleneck (kept at 256 like your 'middle')
        self.mid = Block(base * 4, base * 4, double_conv=use_double_conv)  # 256 -> 256

        # Decoder (ups)
        self.up3 = Up(
            in_ch=base * 4,
            skip_ch=base * 4,
            out_ch=base * 2,
            use_double_conv=use_double_conv,
        )  # 256 + 256 -> 128
        self.up2 = Up(
            in_ch=base * 2,
            skip_ch=base * 2,
            out_ch=base,
            use_double_conv=use_double_conv,
        )  # 128 + 128 -> 64
        self.up1 = Up(
            in_ch=base, skip_ch=base, out_ch=base, use_double_conv=use_double_conv
        )  # 64 + 64   -> 64

        # Head
        self.head = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        # Accept (B, H, W) or (B, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        H, W = x.shape[-2:]

        # Encoder
        x1 = self.inc(x)  # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 256

        # Bottleneck
        xm = self.mid(x4)  # 256

        # Decoder with skips
        x = self.up3(xm, x3)  # -> 128
        x = self.up2(x, x2)  # -> 64
        x = self.up1(x, x1)  # -> 64

        x = self.head(x)  # -> out_ch (e.g., 2)

        # Guarantee exact size match (handles odd sizes / rounding)
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x


def get_model(
    model_type, model_input_channels, unet_double_conv, load_model_path, device
):
    # from local_scripts.assess_csv import percent_error_from_predicted_far_bank_count
    if model_type == "heatmap_cnn":
        model = HeatmapCNN(in_ch=model_input_channels).to(device)
    elif model_type == "unet":
        model = UNetHeatmap(
            in_ch=model_input_channels, use_double_conv=unet_double_conv
        ).to(device)

    if load_model_path:
        # MAH 2025-11-24 12:33:54 TODO this is a hack to get the model to load on the CPU because my machine is showing no GPU available
        try:
            model.load_state_dict(torch.load(load_model_path, weights_only=True))
        except Exception as e:
            print(
                f"MAH TODO this is a hack to get the model to load on the CPU because my machine is showing no GPU available"
            )
            print(f"Error loading model: {e}")

            model.load_state_dict(
                torch.load(
                    load_model_path, weights_only=True, map_location=torch.device("cpu")
                )
            )

        print(f"Loaded model from {load_model_path}")
    else:
        print("No model path provided")
    return model

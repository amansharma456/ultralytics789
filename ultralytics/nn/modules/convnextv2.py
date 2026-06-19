# ultralytics/nn/modules/convnextv2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last and channels_first."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape,
                                self.weight, self.bias, self.eps)
        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GRN(nn.Module):
    """Global Response Normalisation."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """Single ConvNeXtV2 block (depthwise conv + GRN + two pointwise)."""
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm    = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act     = nn.GELU()
        self.grn     = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)          # NCHW → NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)          # NHWC → NCHW
        return shortcut + self.drop_path(x)


class ConvNeXtV2Backbone(nn.Module):
    """
    ConvNeXtV2 used as a YOLOv8 backbone.

    Returns a list of four feature maps at forward():
        index 0 → P2  stride  4
        index 1 → P3  stride  8
        index 2 → P4  stride 16
        index 3 → P5  stride 32

    The YAML file (and parse_model patch) will pick indices 1, 2, 3
    (P3/P4/P5) for the PAN-FPN neck, exactly like the stock backbone.
    """

    arch_settings = {
        # variant : (depths,              dims)
        "atto"  : ([2, 2, 6, 2],  [40,  80,  160,  320]),
        "femto" : ([2, 2, 6, 2],  [48,  96,  192,  384]),
        "pico"  : ([2, 2, 6, 2],  [64,  128, 256,  512]),
        "nano"  : ([2, 2, 8, 2],  [80,  160, 320,  640]),
        "tiny"  : ([3, 3, 9, 3],  [96,  192, 384,  768]),
        "base"  : ([3, 3, 27, 3], [128, 256, 512, 1024]),
        "large" : ([3, 3, 27, 3], [192, 384, 768, 1536]),
        "huge"  : ([3, 3, 27, 3], [352, 704, 1408, 2816]),
    }

    def __init__(self, variant="tiny", in_chans=3, drop_path_rate=0.0):
        super().__init__()
        depths, dims = self.arch_settings[variant]
        self.dims = dims

        # ── downsample layers (stem + 3 inter-stage) ─────────────────────
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        # ── stages ────────────────────────────────────────────────────────
        dp_rates = [x.item() for x in
                    torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dims[i], dp_rates[cur + j])
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # per-stage output norms (channels_first)
        self.norms = nn.ModuleList([
            LayerNorm(d, eps=1e-6, data_format="channels_first")
            for d in dims
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Returns [P2, P3, P4, P5] — all four feature maps."""
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(self.norms[i](x))
        return outs      # list of 4 tensors

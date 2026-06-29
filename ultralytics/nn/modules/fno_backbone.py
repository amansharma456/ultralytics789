"""
FNO Backbone v2 for YOLOv8 — TB Bacilli Detection
Fix: modes are clamped at runtime to H//2 and W//2+1 so the model
     builds correctly at any image size and any YAML scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── ECA ─────────────────────────────────────────────────────────────────────
class ECA(nn.Module):
    """Efficient Channel Attention — parameter-free adaptive kernel."""
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        k = t if t % 2 else t + 1
        self.avg  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x)                                          # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))           # (B,1,C)
        y = y.transpose(-1, -2).unsqueeze(-1)                    # (B,C,1,1)
        return x * self.sig(y)


# ── SpectralConv2d ───────────────────────────────────────────────────────────
class SpectralConv2d(nn.Module):
    """
    2D Fourier layer.
    self.modes1 / self.modes2 are the MAXIMUM modes requested.
    In forward() they are clamped to the actual rfft2 output size so
    the model works at any resolution and never throws a broadcast error.
    """
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1   # max requested
        self.modes2 = modes2   # max requested
        scale = 1.0 / (in_ch * out_ch)
        # weights sized to the MAX modes; at runtime we slice to actual modes
        self.w1 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def _mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── clamp modes to what rfft2 actually gives ───────────────────
        # rfft2 gives (H, W//2+1); upper half is H//2 rows
        # modes1 must be <= H//2 so top and bottom bands do not overlap
        # modes2 must be <= W//2+1
        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)

        x_ft  = torch.fft.rfft2(x, norm="ortho")   # (B, C, H, W//2+1)
        out   = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # slice both weights and the spectrum to the clamped modes
        out[:, :, :m1, :m2] = self._mul(
            x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = self._mul(
            x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])

        return torch.fft.irfft2(out, s=(H, W), norm="ortho")


# ── DropPath ─────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep  = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        r     = torch.rand(shape, device=x.device).floor_() + keep
        return x * r / keep


# ── FNO Block v2 ─────────────────────────────────────────────────────────────
class FNOBlock(nn.Module):
    """
    FNO block with:
      - 3x3 DWConv + 1x1 PW bypass  (local rod-shape texture)
      - ECA channel attention gate   (suppress background)
      - residual add after ECA
    """
    def __init__(self, channels: int, modes1: int, modes2: int,
                 drop_path: float = 0.0):
        super().__init__()
        self.spectral  = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass    = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.norm      = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)
        self.act       = nn.GELU()
        self.eca       = ECA(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm(self.spectral(x) + self.bypass(x)))
        y = self.eca(y)
        return x + self.drop_path(y)


# ── SPPF ─────────────────────────────────────────────────────────────────────
class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast — identical to Ultralytics SPPF."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        h = in_ch // 2
        self.cv1  = nn.Sequential(
            nn.Conv2d(in_ch, h, 1, bias=False),
            nn.BatchNorm2d(h, eps=1e-3, momentum=0.03),
            nn.SiLU(),
        )
        self.cv2  = nn.Sequential(
            nn.Conv2d(h * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03),
            nn.SiLU(),
        )
        self.pool = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y  = self.cv1(x)
        y1 = self.pool(y)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([y, y1, y2, y3], 1))


# ── FNO Backbone v2 ──────────────────────────────────────────────────────────
class FNOBackbone(nn.Module):
    """
    4-block FNO backbone.

    Stem: two 3x3 convs stride-2 each → total stride-4 → P2
    Stages: FNOBlock + ECA at each scale
    P5 gets SPPF for multi-scale global pooling.

    Returns [P2, P3, P4, P5].
    Channel widths [96, 192, 384, 768] match ConvNeXtV2-tiny.
    Modes are MAX values; SpectralConv2d clamps them at runtime.
    """

    _DIMS  = [96,  192, 384, 768]
    _MODES = [20,  16,  12,  8]    # safely below H//2 at 640:
                                   # P2 H//2=80, P3=40, P4=20, P5=10

    def __init__(self,
                 in_chans: int = 3,
                 dims: list | None = None,
                 modes: list | None = None,
                 drop_path_rate: float = 0.1):
        super().__init__()
        dims  = dims  or self._DIMS
        modes = modes or self._MODES
        self.dims = list(dims)

        dp = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]

        # ── Stem ────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[0] // 2, eps=1e-3, momentum=0.03),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[0], eps=1e-3, momentum=0.03),
            nn.GELU(),
        )

        # ── Downsamples ─────────────────────────────────────────────────
        self.downsamples = nn.ModuleList()
        for i in range(4):
            if i == 0:
                self.downsamples.append(nn.Identity())
            else:
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(dims[i-1], dims[i], 2, 2, bias=False),
                    nn.BatchNorm2d(dims[i], eps=1e-3, momentum=0.03),
                ))

        # ── FNO blocks ──────────────────────────────────────────────────
        self.fno_blocks = nn.ModuleList([
            FNOBlock(dims[i], modes[i], modes[i], drop_path=dp[i])
            for i in range(4)
        ])

        # ── SPPF at P5 ──────────────────────────────────────────────────
        self.sppf = SPPF(dims[3], dims[3])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> list:
        x    = self.stem(x)
        outs = []
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.fno_blocks[i](x)
            if i == 3:
                x = self.sppf(x)
            outs.append(x)
        return outs    # [P2_96ch, P3_192ch, P4_384ch, P5_768ch]

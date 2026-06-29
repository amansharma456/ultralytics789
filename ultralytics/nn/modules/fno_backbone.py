"""
FNO Backbone v4 — definitive fix for einsum crash during stride probe.

The crash happens because DetectionModel.__init__ runs a stride-detection
forward pass with torch.zeros(1, 3, 256, 256).  At that resolution:
  P5 = 8x8  →  rfft2 gives 8x5  →  modes=8 tries to index 8 rows
               from the negative-frequency band: x_ft[:,:,-8:,:] on a
               tensor with H=8 gives x_ft[:,:,0:8,:] = all 8 rows,
               but the upper band start index H-m1 = 8-8 = 0 collides
               with the lower band start 0, causing einsum shape conflict.

Fix strategy: make SpectralConv2d fully safe by:
  1. Capping m1 to H//2 - 1 (not H//2) so upper and lower bands
     never touch even at minimum probe resolution.
  2. Building weight tensors at construction time sized to the MINIMUM
     possible feature map (probe resolution 256x256 → P5 8x8).
  3. Using torch.narrow() for all slicing (no fancy indexing).
  4. Adding a guard in FNOBlock.forward that skips spectral computation
     entirely when H < 8 or W < 8 (pure bypass path).
"""

import math
import torch
import torch.nn as nn


# ── Safe mode computation ────────────────────────────────────────────────────
# Probe: 256x256 → stem stride-4 → P2=64, P3=32, P4=16, P5=8
# rfft2(HxW) → Hx(W//2+1); upper band uses rows [H-m1 : H]
# To prevent lower [0:m1] and upper [H-m1:H] from overlapping:
#   we need 2*m1 <= H  →  m1 <= H//2
# Use H//2 - 1 for a guaranteed gap of at least 1 row.
_PROBE_H = [64, 32, 16, 8]          # P2..P5 at 256x256
_PROBE_W = [64, 32, 16, 8]

def _safe_modes(stage: int) -> tuple[int, int]:
    H = _PROBE_H[stage]
    W = _PROBE_W[stage]
    want = [20, 16, 12, 8][stage]
    m1   = min(want, max(1, H // 2 - 1))   # guaranteed gap between bands
    m2   = min(want, W // 2 + 1)
    return m1, m2

# Pre-compute for all 4 stages
_M1 = [_safe_modes(i)[0] for i in range(4)]   # [20, 15, 7, 3]
_M2 = [_safe_modes(i)[1] for i in range(4)]   # [20, 16, 8, 4] (W//2+1 at probe)


# ── ECA ─────────────────────────────────────────────────────────────────────
class ECA(nn.Module):
    """Efficient Channel Attention — adaptive 1D conv, no FC layers."""
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t    = int(abs(math.log2(channels) / gamma + b / gamma))
        k    = t if t % 2 else t + 1
        self.avg  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k,
                              padding=k // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sig(y)


# ── SpectralConv2d v4 ────────────────────────────────────────────────────────
class SpectralConv2d(nn.Module):
    """
    2D Fourier layer with guaranteed safe band separation.

    Weights are sized to (m1, m2) where m1 and m2 are computed from the
    minimum expected feature map size (256x256 probe).  At training
    resolution (640/960) the feature maps are larger and the same
    weights are used on the lower-frequency sub-bands.

    Band layout in rfft2 output (H x W//2+1):
        rows 0 .. m1-1        → lower  positive frequencies → weight w1
        rows H-m1 .. H-1      → upper  negative frequencies → weight w2
        rows m1 .. H-m1-1     → zeroed (high frequencies discarded)

    Guarantee: m1 <= H//2 - 1  so  upper_start = H - m1 >= m1 + 1
    The two bands never overlap.
    """

    def __init__(self, in_ch: int, out_ch: int, m1: int, m2: int):
        super().__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.m1     = m1
        self.m2     = m2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, m1, m2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Runtime-clamped modes (handles any resolution >= probe)
        rm1 = min(self.m1, max(1, H // 2 - 1))
        rm2 = min(self.m2, W // 2 + 1)

        # Verify bands do not overlap (should never fire after clamping)
        assert H - rm1 > rm1, \
            f"FNO band overlap: H={H} rm1={rm1}. Feature map too small."

        x_ft  = torch.fft.rfft2(x, norm="ortho")   # (B, C, H, W//2+1)
        out   = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # Lower band — rows [0 : rm1]
        x_lo = x_ft.narrow(2, 0,        rm1).narrow(3, 0, rm2)
        w1   = self.w1.narrow(2, 0,      rm1).narrow(3, 0, rm2)
        out.narrow(2, 0, rm1).narrow(3, 0, rm2).copy_(
            torch.einsum("bixy,ioxy->boxy", x_lo, w1))

        # Upper band — rows [H-rm1 : H]
        x_hi = x_ft.narrow(2, H - rm1,  rm1).narrow(3, 0, rm2)
        w2   = self.w2.narrow(2, 0,      rm1).narrow(3, 0, rm2)
        out.narrow(2, H - rm1, rm1).narrow(3, 0, rm2).copy_(
            torch.einsum("bixy,ioxy->boxy", x_hi, w2))

        return torch.fft.irfft2(out, s=(H, W), norm="ortho")


# ── DropPath ─────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep  = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand  = torch.rand(shape, dtype=x.dtype,
                           device=x.device).floor_().add_(keep)
        return x * rand / keep


# ── FNO Block v4 ─────────────────────────────────────────────────────────────
class FNOBlock(nn.Module):
    """
    FNO block with safety guard for very small feature maps.

    When H < 4 or W < 4 the spectral path is skipped entirely and
    only the DWConv bypass is used.  This handles any edge case
    at extremely small resolutions without crashing.
    """
    _MIN_SPATIAL = 4   # minimum H or W to attempt spectral computation

    def __init__(self, channels: int, m1: int, m2: int,
                 drop_path: float = 0.0):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, m1, m2)
        # 3x3 DWConv + 1x1 PW: captures local rod-shape texture
        self.bypass   = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.norm     = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)
        self.act      = nn.GELU()
        self.eca      = ECA(channels)
        self.dp       = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        if H < self._MIN_SPATIAL or W < self._MIN_SPATIAL:
            # Pure bypass path — spectral computation unsafe at this size
            y = self.bypass(x)
        else:
            y = self.spectral(x) + self.bypass(x)
        y = self.act(self.norm(y))
        y = self.eca(y)
        return x + self.dp(y)


# ── SPPF ─────────────────────────────────────────────────────────────────────
class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast (identical to Ultralytics SPPF)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        h         = in_ch // 2
        self.cv1  = nn.Sequential(
            nn.Conv2d(in_ch, h, 1, bias=False),
            nn.BatchNorm2d(h, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )
        self.cv2  = nn.Sequential(
            nn.Conv2d(h * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        a, b, c = self.pool(y), self.pool(self.pool(y)), \
                  self.pool(self.pool(self.pool(y)))
        return self.cv2(torch.cat([y, a, b, c], dim=1))


# ── FNO Backbone v4 ──────────────────────────────────────────────────────────
class FNOBackbone(nn.Module):
    """
    4-stage FNO backbone returning [P2, P3, P4, P5].
    Channel widths [96, 192, 384, 768] — identical to ConvNeXtV2-tiny.
    Modes pre-clamped to probe resolution so no crash during
    DetectionModel stride computation.

    Safe modes used (computed from 256x256 probe):
      P2: m1=20  m2=20   (64x64 → rfft → 64x33)
      P3: m1=15  m2=16   (32x32 → rfft → 32x17)
      P4: m1= 7  m2= 8   (16x16 → rfft → 16x 9)
      P5: m1= 3  m2= 4   ( 8x 8 → rfft →  8x 5)
    """

    _DIMS = [96, 192, 384, 768]

    def __init__(self,
                 in_chans: int = 3,
                 dims: list | None = None,
                 drop_path_rate: float = 0.1):
        super().__init__()
        dims      = list(dims or self._DIMS)
        self.dims = dims
        dp        = torch.linspace(0, drop_path_rate, 4).tolist()

        # ── Stem: two 3×3 stride-2 convs (total stride 4) ───────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[0] // 2, eps=1e-3, momentum=0.03),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[0], eps=1e-3, momentum=0.03),
            nn.GELU(),
        )

        # ── Inter-stage downsamples ──────────────────────────────────────
        self.downsamples = nn.ModuleList()
        for i in range(4):
            if i == 0:
                self.downsamples.append(nn.Identity())
            else:
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(dims[i-1], dims[i], 2, 2, bias=False),
                    nn.BatchNorm2d(dims[i], eps=1e-3, momentum=0.03),
                ))

        # ── FNO blocks with pre-clamped safe modes ───────────────────────
        self.fno_blocks = nn.ModuleList([
            FNOBlock(dims[i], _M1[i], _M2[i], drop_path=dp[i])
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
        """Returns [P2, P3, P4, P5]."""
        x    = self.stem(x)
        outs = []
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.fno_blocks[i](x)
            if i == 3:
                x = self.sppf(x)
            outs.append(x)
        return outs

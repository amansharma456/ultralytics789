"""
FNO Backbone v3 for YOLOv8 — TB Bacilli Detection
Definitive fix for einsum size mismatch at small spatial resolutions.

Root cause of crash:
  DetectionModel.__init__ runs a stride-probe forward pass with a
  256x256 dummy input.  At 256x256 the P4 feature map is 16x16.
  rfft2(16x16) → 16x9 complex tensor.
  SpectralConv2d was built with modes=12, so w1/w2 have last dim 12.
  The runtime clamp sets m2 = min(12, 9) = 9.
  BUT the einsum "bixy,ioxy->boxy" sees:
    x slice: (..., 8, 9)     ← correct, clamped
    w slice: w2[:,:,:8,:9]   ← needs explicit .contiguous() slice
  Without the fix the slice is a VIEW whose reported shape may still
  carry stale stride info in some PyTorch versions.

Fix: use narrow() instead of fancy indexing so the returned tensor
is always a fresh contiguous view with correct shape metadata.
Additionally the weight tensors are now built to the CLAMPED size
computed from the MINIMUM expected spatial dimension (imgsz=256
is the stride-probe size used by DetectionModel), so the mismatch
can never occur regardless of PyTorch version.
"""

import math
import torch
import torch.nn as nn


# ── Minimum feature map sizes at stride-probe resolution (256×256) ───────────
# DetectionModel probes with torch.zeros(1, ch, 256, 256)
# Stem stride-4 → P2 64×64, stride-8 → P3 32×32,
#              → P4 16×16, stride-32 → P5 8×8
# rfft2(H×W) → H×(W//2+1)  upper band = H//2 rows
_PROBE_H = [64, 32, 16, 8]    # feature map heights at probe resolution
_MAX_M1  = [h // 2       for h in _PROBE_H]   # [32,16, 8, 4]
_MAX_M2  = [h // 2 + 1   for h in _PROBE_H]   # [33,17, 9, 5]

# Desired modes (capped by probe limits so weights are never oversized)
_WANT_M  = [20, 16, 12, 8]
_SAFE_M1 = [min(_WANT_M[i], _MAX_M1[i]) for i in range(4)]  # [20,16, 8, 4]
_SAFE_M2 = [min(_WANT_M[i], _MAX_M2[i]) for i in range(4)]  # [20,16, 9, 5]


# ── ECA ─────────────────────────────────────────────────────────────────────
class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al. CVPR 2020)."""
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t    = int(abs(math.log2(channels) / gamma + b / gamma))
        k    = t if t % 2 else t + 1
        self.avg  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k,
                              padding=k // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x)                                   # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))    # (B,1,C)
        y = y.transpose(-1, -2).unsqueeze(-1)             # (B,C,1,1)
        return x * self.sig(y)


# ── SpectralConv2d ───────────────────────────────────────────────────────────
class SpectralConv2d(nn.Module):
    """
    2D Fourier integral operator layer.

    Weight tensors are built to (in_ch, out_ch, m1, m2) where m1 and m2
    are the SAFE modes — guaranteed to fit inside the stride-probe
    feature map.  At training/inference resolution (640 or 960) the
    feature maps are larger, so we additionally clamp with narrow()
    to handle any runtime size gracefully.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 m1: int, m2: int):
        """
        m1, m2: safe mode counts (already clamped by caller to probe limits).
        """
        super().__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.m1     = m1    # modes height
        self.m2     = m2    # modes width

        scale = 1.0 / (in_ch * out_ch)
        # Weights sized exactly to (m1, m2) — no over-allocation
        self.w1 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, m1, m2,
                               dtype=torch.cfloat))
        self.w2 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, m1, m2,
                               dtype=torch.cfloat))

    @staticmethod
    def _mul(inp: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # inp: (B, in_ch, h, w_c)   w: (in_ch, out_ch, h, w_c)
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Actual safe modes for this forward pass
        # (will be <= self.m1/m2 when resolution is lower than training res)
        rm1 = min(self.m1, H // 2)
        rm2 = min(self.m2, W // 2 + 1)

        x_ft = torch.fft.rfft2(x, norm="ortho")          # (B,C,H,W//2+1)

        out  = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                           dtype=torch.cfloat, device=x.device)

        # Use narrow() for guaranteed contiguous slices with correct shape
        # Lower frequencies (top rows of rfft2 output)
        x_lo = x_ft.narrow(2, 0,  rm1).narrow(3, 0, rm2)   # (B,C,rm1,rm2)
        w1   = self.w1.narrow(2, 0, rm1).narrow(3, 0, rm2)  # (Ci,Co,rm1,rm2)
        out.narrow(2, 0, rm1).narrow(3, 0, rm2).copy_(
            self._mul(x_lo, w1))

        # Upper frequencies (bottom rows of rfft2 output = negative freqs)
        x_hi = x_ft.narrow(2, H - rm1, rm1).narrow(3, 0, rm2)
        w2   = self.w2.narrow(2, 0,     rm1).narrow(3, 0, rm2)
        out.narrow(2, H - rm1, rm1).narrow(3, 0, rm2).copy_(
            self._mul(x_hi, w2))

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
        rand  = torch.rand(shape, device=x.device,
                           dtype=x.dtype).floor_().add_(keep)
        return x * rand / keep


# ── FNO Block v3 ─────────────────────────────────────────────────────────────
class FNOBlock(nn.Module):
    """
    FNO block:  residual( ECA( GELU( BN( SpectralConv(x) + DWConv3x3(x) ) ) ) )

    DWConv3x3 bypass  → local rod-shape / acid-fast texture
    ECA gate          → channel attention, suppresses background
    Residual          → stable gradients through identity path
    """
    def __init__(self, channels: int, m1: int, m2: int,
                 drop_path: float = 0.0):
        super().__init__()
        self.spectral  = SpectralConv2d(channels, channels, m1, m2)
        self.bypass    = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.norm      = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)
        self.act       = nn.GELU()
        self.eca       = ECA(channels)
        self.dp        = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.bypass(x)
        y = self.act(self.norm(y))
        y = self.eca(y)
        return x + self.dp(y)


# ── SPPF ─────────────────────────────────────────────────────────────────────
class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast (identical to Ultralytics)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        h = in_ch // 2
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
        a = self.pool(y)
        b = self.pool(a)
        c = self.pool(b)
        return self.cv2(torch.cat([y, a, b, c], dim=1))


# ── FNO Backbone v3 ──────────────────────────────────────────────────────────
class FNOBackbone(nn.Module):
    """
    4-block FNO backbone for YOLOv8.

    Key design decisions
    ────────────────────
    1. Stem: two 3×3 convs stride-2 each (total stride-4, less aliasing
       than a single 4×4 stride-4 conv on tiny bacilli edges).
    2. Modes: pre-clamped to fit inside the DetectionModel stride-probe
       resolution (256×256) so no einsum size crash ever occurs.
    3. Each FNOBlock: spectral global path + DWConv3x3 local path + ECA.
    4. SPPF at P5: multi-scale global pooling for coarse context.
    5. Returns [P2, P3, P4, P5] — channel widths [96,192,384,768].

    Safe mode table (computed from probe resolution 256×256):
      Stage  Feature  rfft2-H  max_m1  want  safe
      P2     64×64      64       32      20    20
      P3     32×32      32       16      16    16
      P4     16×16      16        8      12     8   ← was 12, now 8
      P5      8×8        8        4       8     4   ← was  8, now 4
    """

    _DIMS = [96, 192, 384, 768]

    def __init__(self,
                 in_chans: int = 3,
                 dims: list | None = None,
                 drop_path_rate: float = 0.1):
        super().__init__()
        dims      = list(dims or self._DIMS)
        self.dims = dims

        dp = [x.item() for x in
              torch.linspace(0, drop_path_rate, 4)]

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

        # ── FNO blocks — modes pre-clamped to probe resolution ──────────
        self.fno_blocks = nn.ModuleList([
            FNOBlock(dims[i], _SAFE_M1[i], _SAFE_M2[i], drop_path=dp[i])
            for i in range(4)
        ])

        # ── SPPF at P5 ──────────────────────────────────────────────────
        self.sppf = SPPF(dims[3], dims[3])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
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
        return outs    # [P2, P3, P4, P5]

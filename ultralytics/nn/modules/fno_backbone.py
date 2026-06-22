import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    Core FNO layer: FFT → complex weight multiply → iFFT.
    Keeps only the top modes1 × modes2 frequency components.

    Two weight tensors cover the lower and upper frequency bands
    in the height dimension (positive and negative frequencies
    from rfft2), one weight tensor covers the width band.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1   # height frequency modes
        self.modes2 = modes2   # width  frequency modes (rfft half-spectrum)

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels,
                               modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, h, w) complex  |  w: (in_ch, out_ch, h, w) complex
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # rfft2 gives (B, C, H, W//2+1) complex coefficients
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Lower frequency block (top-left of rfft output)
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Upper frequency block (negative height frequencies, bottom-left)
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """
    Single FNO block:
        out = GELU( BN( SpectralConv(x) + W(x) ) )

    W is a 1×1 pointwise conv bypass — keeps the local linear path
    alive so gradients do not have to flow only through the FFT.
    """
    def __init__(self, channels: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass   = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm     = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)
        self.act      = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.bypass(x)))


class FNOBackbone(nn.Module):
    """
    4-block Fourier Neural Operator backbone for YOLOv8.

    Spatial layout (640×640 input)
    ──────────────────────────────
    Stem  4×4 conv stride 4  →  96ch  160×160  (P2)
    FNO Block-1  modes=(20,20)       ← captures ~H/8 periodicity
    Down  2×2 conv stride 2  → 192ch   80×80   (P3)
    FNO Block-2  modes=(16,16)       ← captures ~H/5 periodicity
    Down                     → 384ch   40×40   (P4)
    FNO Block-3  modes=(12,12)       ← captures ~H/3 periodicity
    Down                     → 768ch   20×20   (P5)
    FNO Block-4  modes=( 8, 8)       ← captures ~H/2 periodicity

    Mode selection rationale
    ────────────────────────
    For rfft2 on an H×W tensor the output is H×(W//2+1).
    We need modes1 ≤ H//2 (so upper and lower bands do not overlap)
    and modes2 ≤ W//2+1 (half-spectrum width).

      P2 160×160 : rfft → 160×81  → modes=20  (H//2=80 ✓  W/2+1=81 ✓)
      P3  80×80  : rfft →  80×41  → modes=16  (H//2=40 ✓  W/2+1=41 ✓)
      P4  40×40  : rfft →  40×21  → modes=12  (H//2=20 ✓  W/2+1=21 ✓)
      P5  20×20  : rfft →  20×11  → modes= 8  (H//2=10 ✓  W/2+1=11 ✓)

    Channel widths [96,192,384,768] match ConvNeXtV2-tiny exactly,
    so the PAN-FPN neck YAML needs no changes.

    Returns
    ───────
    list of four tensors [P2, P3, P4, P5]
    """

    # Optimum defaults — do not change unless you change the YAML neck too
    _DEFAULT_DIMS  = [96, 192, 384, 768]
    _DEFAULT_MODES = [20,  16,  12,   8]

    def __init__(self,
                 in_chans: int = 3,
                 dims: list | None = None,
                 modes: list | None = None):
        super().__init__()
        dims  = dims  or self._DEFAULT_DIMS
        modes = modes or self._DEFAULT_MODES
        assert len(dims) == 4 and len(modes) == 4

        self.dims = dims   # expose for parse_model channel tracking

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(dims[0], eps=1e-3, momentum=0.03),
            nn.GELU(),
        )

        # ── Inter-stage downsamples + FNO blocks ──────────────────────────
        self.downsamples = nn.ModuleList()
        self.fno_blocks  = nn.ModuleList()

        for i in range(4):
            if i == 0:
                # Stage 0: stem already downsampled; no extra downsample
                self.downsamples.append(nn.Identity())
            else:
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(dims[i - 1], dims[i],
                              kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(dims[i], eps=1e-3, momentum=0.03),
                ))
            self.fno_blocks.append(FNOBlock(dims[i], modes[i], modes[i]))

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
        # SpectralConv2d weights are already initialised in their own __init__

    def forward(self, x: torch.Tensor) -> list:
        """Returns [P2, P3, P4, P5]."""
        x = self.stem(x)
        outs = []
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.fno_blocks[i](x)
            outs.append(x)
        return outs   # [P2_96ch, P3_192ch, P4_384ch, P5_768ch]

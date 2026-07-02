"""
verify_swin_sod.py — run this before training to confirm everything works
Run: python verify_swin_sod.py
"""

import sys
for k in list(sys.modules.keys()):
    if "ultralytics" in k:
        del sys.modules[k]

import torch
from ultralytics import YOLO


def main():
    model = YOLO(
        "ultralytics/cfg/models/v8/yolov8-swin-p2.yaml"
    )

    m = model.model
    print("Strides:", m.stride)
    # expect tensor([4., 8., 16., 32.])

    m.eval()
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out = m(x)
    print("Output shape:", out.shape)
    # n_anchors = 160^2 + 80^2 + 40^2 + 20^2 = 34000

    total = sum(p.numel() for p in m.parameters())
    print(f"Total params: {total:,}")


if __name__ == "__main__":
    main()

"""
verify_yolo11_c3tr.py — run before training
"""

import sys
for k in list(sys.modules.keys()):
    if "ultralytics" in k:
        del sys.modules[k]

import torch
from ultralytics import YOLO


def main():
    model = YOLO("ultralytics/cfg/models/v11/yolo11-c3tr.yaml")

    m = model.model
    print("Strides:", m.stride)
    # expect tensor([4., 8., 16., 32.])

    m.eval()
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out = m(x)
    print("Output shape:", out.shape)
    # P2=160x160 + P3=80x80 + P4=40x40 + P5=20x20
    # anchors = 25600+6400+1600+400 = 34000
    # expect (1, 5, 34000) for nc=1

    total = sum(p.numel() for p in m.parameters())
    print(f"Total params: {total:,}")


if __name__ == "__main__":
    main()

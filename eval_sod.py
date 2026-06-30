"""
eval_sod.py — Evaluate trained YOLOv8+P2 model
Run: python eval_sod.py --weights runs/tb_sod/yolov8_p2_baseline/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="tb_dataset/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data   = args.data,
        imgsz  = args.imgsz,
        device = args.device,
        split  = "test",
        plots  = True,
        save_json = True,
    )

    print("\n──────────────────────────────────────────")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print("──────────────────────────────────────────")


if __name__ == "__main__":
    main()

"""
predict_c3tr_sod.py
═══════════════════════════════════════════════════════════════════
Run inference with a trained YOLOv8-C3TR-SOD-P2 checkpoint.

Usage:
    # Single image
    python predict_c3tr_sod.py --source path/to/image.jpg

    # All images in a folder
    python predict_c3tr_sod.py --source path/to/images/

    # Video file
    python predict_c3tr_sod.py --source path/to/video.mp4

    # Webcam (device 0)
    python predict_c3tr_sod.py --source 0

    # Custom weights
    python predict_c3tr_sod.py --weights best.pt --source images/
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] pip install ultralytics")


# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Predict with YOLOv8-C3TR-SOD-P2")

    p.add_argument(
        "--weights", type=str,
        default="runs/tb_detect/best.pt",
        help="Path to trained .pt weights"
    )
    p.add_argument(
        "--source", type=str,
        default="tb_dataset/images/test",
        help="Image / directory / video / '0' for webcam"
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size"
    )
    p.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for predictions"
    )
    p.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS"
    )
    p.add_argument(
        "--device", type=str, default="0",
        help="Device: '0' (GPU), 'cpu'"
    )
    p.add_argument(
        "--save", action="store_true", default=True,
        help="Save annotated images/video"
    )
    p.add_argument(
        "--save_txt", action="store_true",
        help="Save YOLO-format labels alongside images"
    )
    p.add_argument(
        "--save_conf", action="store_true",
        help="Include confidence scores in saved txt labels"
    )
    p.add_argument(
        "--show", action="store_true",
        help="Display results window (disable on headless servers)"
    )
    p.add_argument(
        "--project", type=str, default="runs/predict",
        help="Output project directory"
    )
    p.add_argument(
        "--name", type=str, default="c3tr_sod_p2",
        help="Output run name"
    )
    p.add_argument(
        "--max_det", type=int, default=1000,
        help="Maximum detections per image (raise for dense slides)"
    )
    p.add_argument(
        "--tta", action="store_true",
        help="Test-Time Augmentation (slower but more accurate)"
    )
    p.add_argument(
        "--line_width", type=int, default=1,
        help="Bounding box line width in pixels"
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
def auto_find_weights(path: str) -> str:
    """If the specified path does not exist, search runs/ for best.pt."""
    p = Path(path)
    if p.exists():
        return str(p)
    candidates = sorted(Path("runs").rglob("best.pt"))
    if candidates:
        found = str(candidates[-1])
        print(f"[INFO] Auto-found weights: {found}")
        return found
    sys.exit(f"[ERROR] Weights not found: {path}\n"
             "Run train_c3tr_sod.py first, or pass --weights <path>.")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    weights = auto_find_weights(args.weights)

    print(f"\n[INFO] Model    : {weights}")
    print(f"[INFO] Source   : {args.source}")
    print(f"[INFO] Conf     : {args.conf}   IoU: {args.iou}")
    print(f"[INFO] TTA      : {args.tta}")

    model = YOLO(weights)

    # ── Run inference ─────────────────────────────────────────────────────────
    results = model.predict(
        source      = args.source,
        imgsz       = args.imgsz,
        conf        = args.conf,
        iou         = args.iou,
        device      = args.device,
        save        = args.save,
        save_txt    = args.save_txt,
        save_conf   = args.save_conf,
        show        = args.show,
        project     = args.project,
        name        = args.name,
        max_det     = args.max_det,
        augment     = args.tta,
        line_width  = args.line_width,
        verbose     = True,
    )

    # ── Print per-image counts ────────────────────────────────────────────────
    total_boxes = 0
    print(f"\n{'─'*55}")
    print(f"  {'Image':<35s}  {'Detections':>10s}")
    print(f"  {'─'*35}  {'─'*10}")

    for r in results:
        n = len(r.boxes) if r.boxes is not None else 0
        total_boxes += n
        img_name = Path(r.path).name if hasattr(r, "path") else "frame"
        print(f"  {img_name:<35s}  {n:>10d}")

    print(f"{'─'*55}")
    print(f"  {'TOTAL':35s}  {total_boxes:>10d}")
    print(f"{'─'*55}")

    if args.save:
        out_dir = Path(args.project) / args.name
        print(f"\n[INFO] Annotated results saved to: {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

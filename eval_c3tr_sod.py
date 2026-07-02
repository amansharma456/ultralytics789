"""
eval_c3tr_sod.py
═══════════════════════════════════════════════════════════════════
Evaluate a trained YOLOv8-C3TR-SOD-P2 checkpoint.

Usage:
    # Evaluate best weights on both val and test splits
    python eval_c3tr_sod.py --weights runs/tb_detect/<run>/weights/best.pt

    # Also run TTA (Test-Time Augmentation) for maximum mAP
    python eval_c3tr_sod.py --weights best.pt --tta

    # Evaluate on a custom split / different dataset
    python eval_c3tr_sod.py --weights best.pt --data my_data.yaml --split test
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
    p = argparse.ArgumentParser(description="Evaluate YOLOv8-C3TR-SOD-P2")
    p.add_argument(
        "--weights", type=str,
        default="runs/tb_detect/best.pt",
        help="Path to trained .pt weights"
    )
    p.add_argument(
        "--data", type=str,
        default="tb_dataset/data.yaml",
        help="Dataset YAML"
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (px)"
    )
    p.add_argument(
        "--batch", type=int, default=16,
        help="Batch size"
    )
    p.add_argument(
        "--device", type=str, default="0",
        help="Device: '0' for GPU 0, 'cpu'"
    )
    p.add_argument(
        "--conf", type=float, default=0.001,
        help="Confidence threshold (keep low for mAP evaluation)"
    )
    p.add_argument(
        "--iou", type=float, default=0.6,
        help="IoU threshold for NMS"
    )
    p.add_argument(
        "--tta", action="store_true",
        help="Enable Test-Time Augmentation (typically +1-3%% mAP50)"
    )
    p.add_argument(
        "--split", type=str, default="both",
        choices=["val", "test", "both"],
        help="Which split to evaluate"
    )
    p.add_argument(
        "--save_json", action="store_true",
        help="Save COCO-format JSON results"
    )
    p.add_argument(
        "--save_txt", action="store_true",
        help="Save per-image YOLO-format label txt files"
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
def run_eval(model: YOLO, args, split: str) -> dict:
    """Run validation on one split and return metrics dict."""
    print(f"\n{'═'*60}")
    print(f"  Evaluating on  [{split.upper()}]  split")
    if args.tta:
        print("  TTA: ON  (augment=True — runs 3 augmented forward passes)")
    print(f"{'═'*60}")

    metrics = model.val(
        data      = args.data,
        split     = split,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device,
        conf      = args.conf,
        iou       = args.iou,
        augment   = args.tta,          # TTA flag
        verbose   = True,
        save_json = args.save_json,
        save_txt  = args.save_txt,
        plots     = True,              # saves confusion matrix + PR curve
    )

    # ── Pretty-print results ────────────────────────────────────────────────
    print(f"\n  ── {split.upper()} RESULTS {'(TTA)' if args.tta else ''} ──")
    print(f"  {'mAP50':15s}: {metrics.box.map50:.4f}  ({metrics.box.map50*100:.2f} %)")
    print(f"  {'mAP50-95':15s}: {metrics.box.map:.4f}  ({metrics.box.map*100:.2f} %)")
    print(f"  {'Precision':15s}: {metrics.box.mp:.4f}")
    print(f"  {'Recall':15s}: {metrics.box.mr:.4f}")
    print(f"  {'F1':15s}: {2*metrics.box.mp*metrics.box.mr / max(metrics.box.mp+metrics.box.mr, 1e-9):.4f}")

    # ── Target check ────────────────────────────────────────────────────────
    print()
    if metrics.box.map50 >= 0.90:
        print(f"  ✅  TARGET HIT on {split}: mAP50 = {metrics.box.map50*100:.2f} % ≥ 90 %")
    else:
        gap = (0.90 - metrics.box.map50) * 100
        print(f"  ⚠️   {split}: {metrics.box.map50*100:.2f} % — {gap:.1f} pp below 90 %")

    return {
        "split":     split,
        "map50":     metrics.box.map50,
        "map50_95":  metrics.box.map,
        "precision": metrics.box.mp,
        "recall":    metrics.box.mr,
    }


# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # Resolve weights path
    weights = Path(args.weights)
    if not weights.exists():
        # Try common patterns
        candidates = list(Path("runs").rglob("best.pt"))
        if candidates:
            weights = candidates[-1]
            print(f"[INFO] Auto-found weights: {weights}")
        else:
            sys.exit(f"[ERROR] Weights not found: {args.weights}")

    print(f"\n[INFO] Loading model from: {weights}")
    model = YOLO(str(weights))
    model.info(verbose=False)

    results = []

    # ── Evaluate requested splits ────────────────────────────────────────────
    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split in splits:
        r = run_eval(model, args, split)
        results.append(r)

    # ── Summary table ────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'═'*60}")
        print(f"  {'Split':8s}  {'mAP50':>8s}  {'mAP50-95':>10s}  {'P':>8s}  {'R':>8s}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}")
        for r in results:
            print(
                f"  {r['split']:8s}  "
                f"{r['map50']*100:7.2f}%  "
                f"{r['map50_95']*100:9.2f}%  "
                f"{r['precision']*100:7.2f}%  "
                f"{r['recall']*100:7.2f}%"
            )
        print(f"{'═'*60}")

    # ── Save CSV summary ─────────────────────────────────────────────────────
    out_csv = weights.parent.parent / "eval_summary.csv"
    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split","map50","map50_95","precision","recall"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[INFO] Saved summary → {out_csv}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

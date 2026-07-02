"""
train_c3tr_sod.py
═══════════════════════════════════════════════════════════════════
Train  YOLOv8n-C3TR-SOD-P2  for TB bacillus (small object) detection.
Target ≥ 90 mAP50 on the test split.

SETUP (one-time):
─────────────────
1. Copy  yolov8-c3tr-sod-p2.yaml  →  ultralytics/cfg/models/v8/
2. Place this script in the repo root  (same level as ultralytics/)
3. Make sure your dataset YAML at DATA_YAML is correct (see below)
4. Run:
       python train_c3tr_sod.py

DATASET YAML FORMAT  (tb_dataset/data.yaml):
────────────────────────────────────────────
   path: ./tb_dataset          # root
   train: images/train
   val:   images/val
   test:  images/test          # ← must exist for final eval
   nc: 1
   names: ['bacillus']
"""

import os
import sys
import time
import datetime
from pathlib import Path

# ── Sanity-check ultralytics is importable ────────────────────────────────────
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    sys.exit(
        "[ERROR] ultralytics not found.\n"
        "Run:  pip install ultralytics"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  USER SETTINGS  — edit these for your environment
# ══════════════════════════════════════════════════════════════════════════════

# Path to your dataset YAML
DATA_YAML = "tb_dataset/data.yaml"

# Model YAML (relative to repo root; copy the file there first)
MODEL_YAML = "ultralytics/cfg/models/v8/yolov8-c3tr-sod-p2.yaml"

# Which scale to train.  Options: n | s | m | l | x
# Start with 's' — best accuracy-vs-speed for microscopy data.
# Use 'n' if GPU has < 6 GB VRAM.
SCALE = "s"

# Pre-trained weights for backbone initialisation.
# Ultralytics downloads automatically on first run.
# Must match SCALE:  yolov8n.pt / yolov8s.pt / yolov8m.pt …
PRETRAINED = f"yolov8{SCALE}.pt"

# GPU device — 0 for first GPU, 'cpu' if no GPU
DEVICE = 0

# Increase to 960 if you have ≥ 12 GB VRAM (gives +2-3 % mAP on tiny objects)
IMGSZ = 640

# Batch size — reduce to 8 for 960 px or < 8 GB VRAM
BATCH = 16

# Total epochs (cosine LR + early stopping keep this safe to set high)
EPOCHS = 300

# Where results are saved
PROJECT = "runs/tb_detect"
RUN_NAME = f"c3tr_sod_p2_{SCALE}_{datetime.datetime.now():%Y%m%d_%H%M}"

# Resume an interrupted run?  Set to True and point RESUME_WEIGHTS below.
RESUME = False
RESUME_WEIGHTS = "runs/tb_detect/c3tr_sod_p2_s_YYYYMMDD_HHMM/weights/last.pt"

# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
#  (Tuned for small-object microscopy — do not change unless you understand why)
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_ARGS = dict(
    data             = DATA_YAML,
    epochs           = EPOCHS,
    imgsz            = IMGSZ,
    batch            = BATCH,
    device           = DEVICE,
    workers          = 4,

    # ── Optimiser ────────────────────────────────────────────────────────────
    # AdamW is mandatory when C3TR (transformer) blocks are present.
    # SGD diverges in the early epochs because attention weights need
    # adaptive per-parameter LR to settle.
    optimizer        = "AdamW",
    lr0              = 0.001,          # initial learning rate
    lrf              = 0.01,           # final LR = lr0 × lrf  (cosine target)
    momentum         = 0.9,            # β1 for Adam
    weight_decay     = 1e-4,           # L2 regularisation

    # ── LR warmup ────────────────────────────────────────────────────────────
    # 5-epoch warmup is longer than the YOLOv8 default (3) to give the
    # C3TR attention heads time to stabilise before the main LR cycle.
    warmup_epochs    = 5,
    warmup_momentum  = 0.8,
    warmup_bias_lr   = 0.1,

    # ── LR schedule ──────────────────────────────────────────────────────────
    cos_lr           = True,           # cosine annealing; beats step-LR here

    # ── Augmentation ─────────────────────────────────────────────────────────
    #
    # degrees=180:  TB bacilli are rod-shaped and appear at ANY orientation
    #               in ZN-stained slides.  Full rotation invariance is the
    #               single biggest augmentation gain for this task.
    degrees          = 180.0,          # ★ most important aug for rods
    flipud           = 0.5,
    fliplr           = 0.5,
    perspective      = 0.0005,         # subtle slide-warp simulation

    # Colour — ZN staining varies between labs and slide batches
    hsv_h            = 0.02,           # hue shift  ±2 %
    hsv_s            = 0.5,            # saturation ±50 %
    hsv_v            = 0.3,            # value      ±30 %

    scale            = 0.5,            # random scale ±50 %
    shear            = 2.0,            # mild shear

    # Mosaic always on; disable last 30 epochs so model fine-tunes
    # on clean images (avoids artefacts near tile boundaries)
    mosaic           = 1.0,
    close_mosaic     = 30,

    # copy_paste: randomly pastes bacillus instances from other images.
    # Dramatically improves recall on sparse slides (few bacilli / image).
    copy_paste       = 0.3,            # ★ key for rare-object recall

    mixup            = 0.05,           # mild; prevents overconfidence

    # ── Loss weights ─────────────────────────────────────────────────────────
    # box=7.5 is YOLOv8 default — localisation accuracy matters more than
    # classification for a single-class detector, so keep it high.
    box              = 7.5,
    cls              = 0.5,            # low because nc=1
    dfl              = 1.5,

    # ── Early stopping ───────────────────────────────────────────────────────
    patience         = 60,             # stop if no mAP50 improvement for 60 ep

    # ── Pretrained backbone ──────────────────────────────────────────────────
    pretrained       = PRETRAINED,     # ImageNet backbone weights

    # ── Misc ─────────────────────────────────────────────────────────────────
    amp              = True,           # fp16 mixed precision (saves VRAM)
    multi_scale      = 0.0,            # disabled — fixed imgsz is safer here
    seed             = 42,             # reproducibility
    verbose          = True,

    # ── Output ───────────────────────────────────────────────────────────────
    project          = PROJECT,
    name             = RUN_NAME,
    exist_ok         = False,
    save             = True,
    save_period      = 10,             # checkpoint every 10 epochs
)


# ══════════════════════════════════════════════════════════════════════════════
def check_dataset(yaml_path: str) -> None:
    """Verify the dataset YAML and key split directories exist."""
    p = Path(yaml_path)
    if not p.exists():
        sys.exit(
            f"[ERROR] Dataset YAML not found: {yaml_path}\n"
            "Create tb_dataset/data.yaml with train/val/test splits."
        )

    import yaml
    with open(p) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", p.parent))
    for split in ("train", "val", "test"):
        split_dir = root / cfg.get(split, split)
        if not split_dir.exists():
            print(f"[WARN] Split directory not found: {split_dir}")
        else:
            n_imgs = len(list(split_dir.glob("*.jpg")) +
                         list(split_dir.glob("*.png")) +
                         list(split_dir.glob("*.bmp")))
            print(f"  {split:6s}: {n_imgs:5d} images  ({split_dir})")


# ══════════════════════════════════════════════════════════════════════════════
def build_model() -> YOLO:
    """Build or resume the model."""
    if RESUME:
        print(f"\n[INFO] Resuming from: {RESUME_WEIGHTS}")
        return YOLO(RESUME_WEIGHTS)

    model = YOLO(MODEL_YAML)
    # Print a compact architecture summary
    model.info(verbose=False)
    return model


# ══════════════════════════════════════════════════════════════════════════════
def train(model: YOLO):
    """Run training."""
    print("\n" + "═" * 65)
    print("  YOLOv8-C3TR-SOD-P2   ·   Training")
    print("═" * 65)
    print(f"  Scale      : {SCALE}")
    print(f"  Dataset    : {DATA_YAML}")
    print(f"  Image size : {IMGSZ} px")
    print(f"  Batch      : {BATCH}")
    print(f"  Epochs     : {EPOCHS}  (patience={TRAIN_ARGS['patience']})")
    print(f"  Device     : {DEVICE}")
    print(f"  Output     : {PROJECT}/{RUN_NAME}")
    print("═" * 65 + "\n")

    t0 = time.time()
    results = model.train(**TRAIN_ARGS)
    elapsed = str(datetime.timedelta(seconds=int(time.time() - t0)))

    print(f"\n[INFO] Training complete in {elapsed}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model: YOLO) -> None:
    """
    Evaluate best checkpoint on the TEST split.
    This is the final number to report.
    """
    print("\n" + "═" * 65)
    print("  Final Evaluation on TEST split")
    print("═" * 65)

    metrics = model.val(
        data    = DATA_YAML,
        split   = "test",          # ← test, not val
        imgsz   = IMGSZ,
        batch   = BATCH,
        device  = DEVICE,
        verbose = True,
        save_json = True,          # saves coco-format results.json
    )

    print("\n" + "─" * 40)
    print(f"  mAP50      : {metrics.box.map50:.4f}  ({metrics.box.map50*100:.2f} %)")
    print(f"  mAP50-95   : {metrics.box.map:.4f}  ({metrics.box.map*100:.2f} %)")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print("─" * 40)

    if metrics.box.map50 >= 0.90:
        print("  ✅  TARGET ACHIEVED: mAP50 ≥ 90 %")
    else:
        gap = (0.90 - metrics.box.map50) * 100
        print(f"  ⚠️   {gap:.1f} pp below target.  See README for next steps.")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n[INFO] Checking dataset …")
    check_dataset(DATA_YAML)

    model = build_model()
    train(model)

    # Load best weights for evaluation
    best_pt = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n[INFO] Loading best weights: {best_pt}")
        model = YOLO(str(best_pt))
    else:
        print("[WARN] best.pt not found — evaluating with current model state")

    evaluate(model)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

"""
tb_fno_train.py
Run:  python tb_fno_train.py
"""

from ultralytics import YOLO
from ultralytics.nn.tasks import DualDetectionModel


def main():
    # Build the dual-head FNO model
    model = YOLO("ultralytics/cfg/models/v8/yolov8-fno-dual.yaml",
                 task="detect")
    # Swap internal model class so DualDetectionModel.__init__ runs
    model.model = DualDetectionModel(
        "ultralytics/cfg/models/v8/yolov8-fno-dual.yaml", nc=1
    )

    model.train(
        data       = "tb_dataset/data.yaml",
        epochs     = 200,
        # ── Resolution ─────────────────────────────────────────────────
        # 960 instead of 640. At stride-4 (P2) a 4px bacillus becomes
        # a 1px feature at 640 but a 1.5px feature at 960 — meaningful
        # for the FNO modes at P2 (modes=20 × 960/640 = 30 effective modes).
        # Compensate for larger batch with gradient accumulation.
        imgsz      = 960,
        batch      = 8,              # reduce from 16 due to 960 resolution
        # ── Gradient accumulation ────────────────────────────────────────
        # Effective batch = 8 × 2 = 16, same as before but at 960
        # Ultralytics uses nbs (nominal batch size) for auto LR scaling
        # accumulate = 2 is set via override below

        # ── Optimiser ────────────────────────────────────────────────────
        optimizer  = "AdamW",
        lr0        = 5e-4,           # lower than default (1e-2 for SGD)
        lrf        = 0.01,           # final LR = lr0 * lrf = 5e-6
        momentum   = 0.9,            # AdamW beta1
        weight_decay = 1e-4,

        # ── LR schedule ──────────────────────────────────────────────────
        # cosine annealing (Ultralytics default when cos_lr=True)
        cos_lr     = True,
        warmup_epochs = 10,          # longer warmup for FNO (spectral weights)
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.1,

        # ── EMA ─────────────────────────────────────────────────────────
        # Exponential moving average — critical for small datasets
        # Higher decay = smoother EMA = better generalisation
        # Default is 0.9999; keep it
        # (set via model.trainer.ema.decay if needed)

        # ── Augmentation ──────────────────────────────────────────────────
        # Microscopy-specific: heavy spatial, moderate colour
        degrees    = 15.0,           # bacilli rotate freely
        translate  = 0.1,
        scale      = 0.5,
        shear      = 2.0,
        perspective= 0.0,
        flipud     = 0.5,            # no canonical up in microscopy
        fliplr     = 0.5,
        mosaic     = 1.0,
        mixup      = 0.05,
        copy_paste = 0.4,            # copy individual bacilli to sparse slides
        hsv_h      = 0.02,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        erasing    = 0.3,

        # ── Loss weights ──────────────────────────────────────────────────
        box        = 7.5,
        cls        = 0.5,
        dfl        = 1.5,

        # ── Task Aligned Assigner tuning ──────────────────────────────────
        # topk=10 (default=10), increase to 15 for dense small objects
        # alpha controls classification alignment, beta controls iou alignment
        # Higher beta forces more precise localisation
        # Set via override; Ultralytics reads these from cfg
        tal_topk   = 15,
        tal_alpha  = 0.5,
        tal_beta   = 6.0,

        # ── Misc ───────────────────────────────────────────────────────────
        device     = 0,
        workers    = 4,
        project    = "runs/tb_fno_v2",
        name       = "dual_head_960",
        save       = True,
        val        = True,
        plots      = True,
        verbose    = True,
        patience   = 40,             # early stopping — longer for slow FNO convergence
        amp        = True,           # mixed precision — essential at 960 resolution
        close_mosaic = 20,           # disable mosaic in last 20 epochs for fine-tuning
        nbs        = 64,             # nominal batch size → auto-scales LR
                                     # effective_lr = lr0 * (batch/nbs)^0.5
                                     # with batch=8: lr = 5e-4 * sqrt(8/64) = 1.77e-4
    )


if __name__ == "__main__":
    main()

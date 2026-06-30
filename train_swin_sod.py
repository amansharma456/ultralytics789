"""
train_swin_sod.py — YOLOv8 + Swin-T backbone + P2 SOD
Run: python train_swin_sod.py
"""

from ultralytics import YOLO


def main():
    model = YOLO("ultralytics/cfg/models/v8/yolov8-swin-p2.yaml")

    model.train(
        data        = "tb_dataset/data.yaml",
        epochs      = 200,            # Transformers converge slower — more epochs
        imgsz       = 640,
        batch       = 8,              # reduce — Swin attention is memory-heavier than CNN
        device      = 0,
        workers     = 4,

        # ── optimiser — Transformer backbones need AdamW, not SGD ───────
        optimizer   = "AdamW",
        lr0         = 5e-4,
        lrf         = 0.01,
        weight_decay= 5e-2,           # Swin paper uses higher WD than CNNs
        warmup_epochs = 10,           # longer warmup — attention weights are unstable early
        warmup_momentum = 0.8,
        cos_lr      = True,

        # ── loss weights ─────────────────────────────────────────────────
        box         = 7.5,
        cls         = 0.5,
        dfl         = 1.5,

        # ── augmentation ────────────────────────────────────────────────
        degrees     = 15.0,
        translate   = 0.1,
        scale       = 0.5,
        shear       = 2.0,
        flipud      = 0.5,
        fliplr      = 0.5,
        mosaic      = 1.0,
        mixup       = 0.05,
        copy_paste  = 0.3,
        hsv_h       = 0.02,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        close_mosaic= 20,

        project     = "runs/tb_swin_sod",
        name        = "swin_p2",
        exist_ok    = True,
        save        = True,
        val         = True,
        plots       = True,
        verbose     = True,
        patience    = 40,             # slower convergence needs more patience
        amp         = True,
    )


if __name__ == "__main__":
    main()

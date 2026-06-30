"""
train_sod.py — YOLOv8 baseline + P2 Small Object Detection
Run: python train_sod.py
"""

from ultralytics import YOLO


def main():
    model = YOLO("ultralytics/cfg/models/v8/yolov8-p2.yaml")

    model.train(
        data        = "tb_dataset/data.yaml",
        epochs      = 150,
        imgsz       = 640,
        batch       = 16,
        device      = 0,
        workers     = 4,

        # ── optimiser ──────────────────────────────────────────────────
        optimizer   = "SGD",
        lr0         = 0.01,
        lrf         = 0.01,
        momentum    = 0.937,
        weight_decay= 5e-4,
        warmup_epochs = 3,

        # ── loss weights ──────────────────────────────────────────────
        # box weight raised slightly: P2 head needs precise small-box
        # regression; cls/dfl left at defaults
        box         = 8.0,
        cls         = 0.5,
        dfl         = 1.5,

        # ── augmentation tuned for microscopy ────────────────────────
        degrees     = 15.0,
        translate   = 0.1,
        scale       = 0.5,
        shear       = 2.0,
        perspective = 0.0,
        flipud      = 0.5,   # no canonical orientation in microscopy
        fliplr      = 0.5,
        mosaic      = 1.0,
        mixup       = 0.05,
        copy_paste  = 0.3,   # helps with sparse bacilli
        hsv_h       = 0.02,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        close_mosaic= 15,    # disable mosaic for last 15 epochs

        project     = "runs/tb_sod",
        name        = "yolov8_p2_baseline",
        exist_ok    = True,
        save        = True,
        val         = True,
        plots       = True,
        verbose     = True,
        patience    = 30,
        amp         = True,
    )


if __name__ == "__main__":
    main()

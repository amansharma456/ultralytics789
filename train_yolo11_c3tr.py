"""
train_yolo11_c3tr.py
YOLO11 + C3TR at P5 + SOD (P2 head)
Run: python train_yolo11_c3tr.py
"""

from ultralytics import YOLO


def main():
    model = YOLO("ultralytics/cfg/models/v11/yolo11-c3tr.yaml")

    model.train(
        data          = "tb_dataset/data.yaml",
        epochs        = 150,
        imgsz         = 640,
        batch         = 16,
        device        = 0,
        workers       = 4,

        optimizer     = "SGD",
        lr0           = 0.01,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 5e-4,
        warmup_epochs = 3,
        cos_lr        = True,

        box           = 7.5,
        cls           = 0.5,
        dfl           = 1.5,

        degrees       = 15.0,
        translate     = 0.1,
        scale         = 0.5,
        shear         = 2.0,
        flipud        = 0.5,
        fliplr        = 0.5,
        mosaic        = 1.0,
        mixup         = 0.05,
        copy_paste    = 0.3,
        hsv_h         = 0.02,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        close_mosaic  = 15,

        project       = "runs/tb_yolo11_c3tr",
        name          = "yolo11_c3tr_sod",
        exist_ok      = True,
        save          = True,
        val           = True,
        plots         = True,
        verbose       = True,
        patience      = 30,
        amp           = True,
    )


if __name__ == "__main__":
    main()

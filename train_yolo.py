from pathlib import Path
import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "dataset" / "data.yaml"

MODEL_NAME = "yolo11s.pt"
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE =2
DEVICE = 0


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML was not found:\n{DATA_YAML}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install the CUDA-enabled PyTorch build first."
        )

    print("=" * 60)
    print("YOLO11 CUDA WASTE TRAINING")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"GPU VRAM: {torch.cuda.get_device_properties(DEVICE).total_memory / 1024**3:.1f} GB")
    print(f"Dataset: {DATA_YAML}")
    print("=" * 60)

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        pretrained=True,

        device=DEVICE,
        batch=BATCH_SIZE,
        workers=0,
        amp=True,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,

        patience=10,
        val=True,
        save=True,
        save_period=10,
        plots=True,

        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,
        degrees=8.0,
        translate=0.10,
        scale=0.50,
        shear=2.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.10,
        close_mosaic=20,

        seed=42,
        deterministic=True,

        project=str(PROJECT_DIR / "runs"),
        name="waste_yolo11s_cuda",
        exist_ok=True,
        verbose=True
    )


if __name__ == "__main__":
    main()
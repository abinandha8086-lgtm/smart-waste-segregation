from pathlib import Path
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent

LAST_CHECKPOINT = (
    PROJECT_DIR
    / "runs"
    / "waste_yolo11s_cuda"
    / "weights"
    / "last.pt"
)

if not LAST_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{LAST_CHECKPOINT}"
    )

print(f"Resuming from:\n{LAST_CHECKPOINT}")

model = YOLO(str(LAST_CHECKPOINT))

model.train(
    resume=True
)
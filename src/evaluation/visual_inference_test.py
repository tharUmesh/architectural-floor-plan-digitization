from pathlib import Path
from ultralytics import YOLO

# Resolve project root (two levels up from src/evaluation/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

model = YOLO(str(PROJECT_ROOT / "runs/detect/models/yolo11l-2/weights/best.pt"))

# Predict on the entire test folder
results = model.predict(
    source=str(PROJECT_ROOT / "data/yolo_dataset_processed/images/test"),
    conf=0.25,  # Minimum confidence to accept a detection
    iou=0.45,  # NMS threshold to clear overlapping duplicate boxes
    save=True,  # Saves annotated images
    project=str(
        PROJECT_ROOT / "runs/detect"
    ),  # Absolute path to avoid global settings override
    name="predict",  # Subfolder name
    device=0,
)
print(f"Visual predictions saved to {PROJECT_ROOT / 'runs/detect/predict/'}")

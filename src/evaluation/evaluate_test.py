from ultralytics import YOLO


# ── Which model to evaluate ──────────────────────────────────────────────────
# Change this path when evaluating different training runs
MODEL_WEIGHTS = "runs/detect/models/11m_final/weights/best.pt"

# Class names for per-class reporting
CLASS_NAMES = {
    0: "Door",
    1: "Window",
    2: "Wall",
    3: "Staircase",
    4: "Toilet",
    5: "Sink",
}


def main():
    model = YOLO(MODEL_WEIGHTS)

    print(f"Evaluating: {MODEL_WEIGHTS}")
    print("Running final evaluation on the unseen TEST split...")
    metrics = model.val(data="data/yolo_dataset_processed/dataset.yaml", split="test")

    # Print overall metrics
    print("\n================ FINAL TEST METRICS ================")
    print(f"Precision (P):       {metrics.box.mp:.4f}")
    print(f"Recall (R):          {metrics.box.mr:.4f}")
    print(f"mAP50:               {metrics.box.map50:.4f}")
    print(f"mAP50-95:            {metrics.box.map:.4f}")
    print("====================================================")

    # Print per-class AP breakdown
    print("\n============== PER-CLASS AP50 BREAKDOWN =============")
    ap50_per_class = metrics.box.ap50
    for i, ap in enumerate(ap50_per_class):
        name = CLASS_NAMES.get(i, f"Class_{i}")
        print(f"  {name:<12}  AP50: {ap:.4f}")
    print("====================================================")


if __name__ == "__main__":
    main()

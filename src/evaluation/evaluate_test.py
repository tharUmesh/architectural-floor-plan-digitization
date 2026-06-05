from ultralytics import YOLO


def main():
    # Load your freshly trained Large model weights
    model = YOLO("runs/detect/models/yolo11l-2/weights/best.pt")

    print("Running final evaluation on the unseen TEST split...")
    # Force YOLO to evaluate specifically on the 'test' split listed in dataset.yaml
    metrics = model.val(data="data/yolo_dataset_processed/dataset.yaml", split="test")

    # Print critical metrics for your final report
    print("\n================ FINAL TEST METRICS ================")
    print(f"Precision (P):       {metrics.box.mp:.4f}")
    print(f"Recall (R):          {metrics.box.mr:.4f}")
    print(f"mAP50:               {metrics.box.map50:.4f}")
    print(f"mAP50-95:            {metrics.box.map:.4f}")
    print("====================================================")


if __name__ == "__main__":
    main()

"""
YOLOv11 Training Script — Phase 3 (Ubuntu 16GB VRAM Optimized)

HOW TO RUN:
    python src/models/train_yolo.py
"""

import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    # Load the YOLOv11 Small pre-trained model
    print("Loading YOLOv11l model...")
    model = YOLO("yolo11l.pt")

    # Start training with hardware-specific constraints
    print("Starting training loop...")
    results = model.train(
        data="dataset.yaml",
        epochs=150,
        imgsz=1024,
        # --- HARDWARE EXPLOITATION (16GB VRAM) ---
        batch=16,  # Huge bump from 4. If you somehow hit OOM, drop to 8.
        device=0,  # Force use of the primary GPU
        workers=8,  # Increased CPU workers to feed the GPU faster
        amp=True,  # Automatic Mixed Precision (speeds up RTX training)
        # --- THE FIXES ---
        image_weights=True,  # Forces model to oversample rare classes (Sinks/Toilets)
        box=10.0,  # Increase Box loss gain (default is 7.5, push it to 10.0 if still failing)
        cls=0.5,  # Class loss gain
        # --- AUGMENTATION STRATEGY ---
        mosaic=1.0,  # Combines 4 floor plans into one (scale variance)
        degrees=90.0,  # Floor plans can be oriented anywhere
        fliplr=0.5,  # Horizontal flips are structurally valid
        flipud=0.5,  # Vertical flips are structurally valid
        # --- DISABLED AUGMENTATIONS (Irrelevant for blueprints) ---
        # hsv_h=0.0,  # No color hue shifts
        # hsv_s=0.0,  # No color saturation shifts
        # hsv_v=0.0,  # No brightness shifts
        # erasing=0.0,  # Prevent the model from masking out small fixtures like sinks
        # --- LOGGING & SAVING ---
        project="models",
        name="yolo11l",
        save=True,
        save_period=10,  # Save a checkpoint every 10 epochs
    )

    print("Training complete! Best weights are saved in models/yolo11l/weights/")


if __name__ == "__main__":
    main()

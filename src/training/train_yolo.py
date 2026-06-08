"""
YOLOv11 Training Script — Phase 3 (Ubuntu 16GB VRAM Optimized)
VERSION: Step 1 — Augmentation & Hyperparameter Fix
CHANGES FROM BASELINE:
  - Enabled geometric augmentations (mosaic, rotation, flips)
  - Lowered learning rate (0.01 → 0.001) for better fine-tuning
  - Added cosine LR schedule for smoother convergence
  - Longer warmup (3 → 5 epochs)
  - Tighter early stopping (100 → 40 epochs patience)
  - Slight brightness augmentation (hsv_v=0.1)
  - Added translate=0.2 for position variance
VERSION: Step 2 — Larger Model + Extended Training
CHANGES FROM STEP 1 (mAP50=0.79):
  - Upgraded model: YOLO11m → YOLO11l (more capacity, dataset is large enough)
  - Lowered LR: 0.001 → 0.0008 for finer convergence
  - Added mixup=0.15 (blends two images — improves class boundary learning)
  - Increased epochs: 200 → 300 (more room to converge)
  - Extended patience: 40 → 50 (allow model more exploration time)
  - Extended close_mosaic: 20 → 30 (longer fine-tuning without mosaic)

HOW TO RUN:
    python src/training/train_yolo.py
"""

import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    # Load the YOLOv11 Large pre-trained model
    # With ~3500 training images, the larger model can leverage its capacity
    print("Loading YOLOv11l model...")
    model = YOLO("yolo11l.pt")

    # Start training with hardware-specific constraints
    print("Starting training loop...")
    results = model.train(
        data="data/yolo_dataset_processed/dataset.yaml",
        epochs=200,
        imgsz=1024,
        # --- HARDWARE EXPLOITATION (16GB VRAM) ---
        batch=8,  # Reduced from 16 for larger model — prevents OOM at 1024px
        device=0,  # Force use of the primary GPU
        workers=8,  # CPU workers to feed the GPU faster
        amp=True,  # Automatic Mixed Precision (speeds up RTX training)
        # --- LEARNING RATE (slightly lower for finer convergence) ---
        lr0=0.0008,  # Slightly lower than Step 1 for more stable large-model training
        lrf=0.01,  # Final LR = lr0 * lrf
        cos_lr=True,  # Cosine annealing — smoother convergence to better minima
        warmup_epochs=5,  # Warmup for stability
        # --- AUGMENTATION STRATEGY ---
        mosaic=1.0,  # Combines 4 floor plans into one
        close_mosaic=30,  # Turn off mosaic for last 30 epochs for fine-tuning
        mixup=0.15,  # NEW: Blend two images — helps model learn class boundaries
        degrees=90.0,  # Floor plans can be oriented in any direction
        fliplr=0.5,  # Horizontal flips are structurally valid
        flipud=0.5,  # Vertical flips are structurally valid
        scale=0.5,  # Scale variation ±50%
        translate=0.2,  # Position variation ±20%
        # --- DISABLED AUGMENTATIONS (Irrelevant for blueprints) ---
        hsv_h=0.0,  # No color hue shifts (floor plans are grayscale)
        hsv_s=0.0,  # No color saturation shifts
        hsv_v=0.1,  # Slight brightness variation to handle scan quality
        erasing=0.0,  # Prevent masking out small fixtures like sinks
        # --- EARLY STOPPING ---
        patience=50,  # Stop if no improvement for 50 epochs
        # --- LOGGING & SAVING ---
        project="models",
        name="yolo11l_step2",
        save=True,
        save_period=10,  # Save a checkpoint every 10 epochs
    )

    print("Training complete! Best weights are saved in models/yolo11l_step2/weights/")


if __name__ == "__main__":
    main()


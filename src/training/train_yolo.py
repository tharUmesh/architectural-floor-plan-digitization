"""
YOLOv11 Training Script — Phase 3 (RTX 5090 32GB VRAM Optimized)

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
  - Upgraded model: YOLO11m → YOLO11l (more capacity)
  - Color-preserving preprocessing (LAB CLAHE + bilateral filter)
  - Lowered LR: 0.001 → 0.0008 for finer convergence
  - Added mixup=0.15 (image blending for class boundaries)
  - Increased epochs: 200 → 300 (more room to converge)
  - Extended patience: 40 → 50 (allow model more exploration time)
  - Extended close_mosaic: 20 → 30 (longer fine-tuning without mosaic)
  - Added copy_paste=0.15 (copies objects between images — helps rare classes)
  - Increased cls loss weight: 0.5 → 1.5 (forces better classification)
  - Reduced scale: 0.5 → 0.3 (prevents shrinking small fixtures)


HOW TO RUN:
    python src/training/train_yolo.py

CUDA FIX (if torch can't see GPU):
    uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
"""

import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    # --- CUDA pre-check ---
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        print(f"  torch version:  {torch.__version__}")
        print(f"  CUDA compiled:  {torch.version.cuda}")
        print()
        print("Fix: uv pip install --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
        sys.exit(1)
    print(f"CUDA OK: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    # Load the YOLOv11 Medium pre-trained model
    print("Loading YOLOv26m model...")
    model = YOLO("yolo26m.pt")

    # Start training with hardware-specific constraints
    print("Starting training loop...")
    results = model.train(
        data="data/yolo_dataset_processed/dataset.yaml",
        epochs=150,
        imgsz=1280,  # Higher resolution for better small object detection
        # --- HARDWARE EXPLOITATION (RTX 5090 32GB VRAM) ---
        batch=8,  # 32GB VRAM can handle batch=32 at 1024px; reduce to 16 if OOM
        device=0,  # Force use of the primary GPU
        workers=8,  # CPU workers to feed the GPU faster
        amp=True,  # Automatic Mixed Precision (speeds up RTX training)
        # --- LEARNING RATE ---
        lr0=0.0008,  # Fine-tuning rate for pretrained large model
        lrf=0.01,  # Final LR = lr0 * lrf
        cos_lr=True,  # Cosine annealing — smoother convergence
        warmup_epochs=5,  # Warmup for stability
        # --- LOSS FUNCTION WEIGHTS ---
        cls=1.5,  # ↑ from 0.5 default — forces model to distinguish classes better
        #           Targets Toilet (0.665) and Staircase (0.725) weak AP
        box=7.5,  # ↓ from 10.0 default — slight rebalance toward classification
        # --- AUGMENTATION STRATEGY ---
        mosaic=1.0,  # Combines 4 floor plans into one
        close_mosaic=15,  # Turn off mosaic for last 15 epochs for fine-tuning
        #mixup=0.15,  # Blend two images — helps class boundary learning
        copy_paste=0.3,  # NEW: Copy objects between images — boosts rare classes
        #degrees=90.0,  # Floor plans can be oriented in any direction
        fliplr=0.5,  # Horizontal flips are structurally valid
        flipud=0.5,  # Vertical flips are structurally valid
        scale=0.3,  # ↓ from 0.5 — less extreme to avoid shrinking small fixtures
        translate=0.2,  # Position variation ±20%
        # --- DISABLED AUGMENTATIONS (Irrelevant for blueprints) ---
        hsv_h=0.0,  # No color hue shifts (floor plans are grayscale)
        hsv_s=0.0,  # No color saturation shifts
        hsv_v=0.1,  # Slight brightness variation to handle scan quality
        erasing=0.0,  # Prevent masking out small fixtures like sinks
        # --- EARLY STOPPING ---
        patience=40,  # Stop if no improvement for 40 epochs
        # --- LOGGING & SAVING ---
        project="models",
        name="yolo26m_final",
        save=True,
        save_period=10,  # Save a checkpoint every 10 epochs
    )

    print("Training complete! Best weights are saved in models/yolo26m_final/weights/")


if __name__ == "__main__":
    main()

"""
Annotation Verification Script — Phase 2

PURPOSE:
    Visually verify that converted YOLO annotations are correctly
    positioned on their images. Draws colored bounding boxes on a
    random sample of training images.

    Run AFTER convert_annotations.py.
    Check the output images in data/verify/ before proceeding to Phase 3.

HOW TO RUN:
    make verify
    OR
    python src/data/verify_annotations.py
"""

import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config, load_env
from src.utils.logger import get_logger

logger = get_logger(__name__)

# One distinct color per class (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0,   255,   0),   # Door      → green
    1: (255,   0,   0),   # Window    → blue
    2: (0,     0, 255),   # Wall      → red
    3: (0,   255, 255),   # Staircase → yellow
    4: (255,   0, 255),   # Toilet    → magenta
    5: (255, 128,   0),   # Sink      → orange
}

VERIFY_SAMPLES = 20   # how many images to verify
RANDOM_SEED    = 99   # different seed from selection to get varied samples


def draw_annotations(img_path: Path, label_path: Path, class_names: dict) -> None:
    """
    Draw bounding boxes from a YOLO label file onto the corresponding image.
    Saves the annotated image to data/verify/.

    The bounding boxes are denormalized using actual image dimensions,
    which should match the dimensions used during annotation conversion.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"Could not read image: {img_path}")
        return

    h, w = img.shape[:2]

    if not label_path.exists():
        logger.warning(f"No label file for: {img_path.name}")
        return

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        logger.warning(f"Empty label file: {label_path.name}")
        return

    annotation_count = 0

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        try:
            cls_id          = int(parts[0])
            x_c, y_c, bw, bh = map(float, parts[1:])

            # Convert normalized YOLO coords back to pixel coords
            # These use the actual image dimensions (width, height)
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            color     = CLASS_COLORS.get(cls_id, (128, 128, 128))
            cls_name  = class_names.get(cls_id, str(cls_id))

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw class label above the box
            cv2.putText(
                img, cls_name, (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
            )

            annotation_count += 1

        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing line '{line}': {e}")
            continue

    # ── Save to verify directory ──────────────────────────────────────────
    verify_dir = Path("data/verify")
    verify_dir.mkdir(exist_ok=True)
    out_path = verify_dir / img_path.name

    cv2.imwrite(str(out_path), img)

    logger.debug(f"Verified {img_path.name}: {annotation_count} annotations drawn")


def main():
    load_env()

    dataset_config = load_config("configs/dataset.yaml")
    class_names    = {
        int(idx): name
        for idx, name in dataset_config["classes"].items()
    }

    # Gather all training images that have a corresponding label
    images_dir = Path("data/yolo_dataset/images/train")
    labels_dir = Path("data/yolo_dataset/labels/train")

    if not images_dir.exists():
        logger.error(
            f"Images directory not found: {images_dir}\n"
            "Run src/data/convert_annotations.py first."
        )
        sys.exit(1)

    all_images = sorted(images_dir.glob("*.png"))

    if not all_images:
        logger.error(
            f"No images found in {images_dir}\n"
            "Run src/data/convert_annotations.py first."
        )
        sys.exit(1)

    # Filter to only images with corresponding label files
    paired = [
        img for img in all_images
        if (labels_dir / (img.stem + ".txt")).exists()
    ]

    if not paired:
        logger.error(
            f"No image+label pairs found in {images_dir}\n"
            "Run src/data/convert_annotations.py first."
        )
        sys.exit(1)

    random.seed(RANDOM_SEED)
    sample = random.sample(paired, min(VERIFY_SAMPLES, len(paired)))

    logger.info(f"Verifying {len(sample)} random training images...")

    for img_path in sample:
        label_path = labels_dir / (img_path.stem + ".txt")
        draw_annotations(img_path, label_path, class_names)

    verify_dir = Path("data/verify")
    logger.info(f"\n✓ Verification complete!")
    logger.info(f"  Annotated images saved to: {verify_dir}/")
    logger.info(f"  Sample count: {len(sample)}")

    print("\n" + "=" * 60)
    print("ANNOTATION VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"\nGenerated {len(sample)} annotated images in {verify_dir}/\n")
    print("Check the images and confirm:")
    print("  ✓ Green boxes around door openings")
    print("  ✓ Blue boxes around windows")
    print("  ✓ Red boxes around wall segments")
    print("  ✓ Yellow boxes around staircases")
    print("  ✓ Magenta boxes around toilets")
    print("  ✓ Orange boxes around sinks")
    print("  ✓ Boxes are correctly sized and positioned")
    print("  ✓ Boxes align with image features (not off by pixels)")
    print("  ✓ No boxes covering the wrong elements")
    print("\nIf alignment is correct, proceed to Phase 3:")
    print("  python src/preprocessing/preprocess_pipeline.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
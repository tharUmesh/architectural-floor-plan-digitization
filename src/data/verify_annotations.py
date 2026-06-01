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

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id          = int(parts[0])
        x_c, y_c, bw, bh = map(float, parts[1:])

        # Convert normalized YOLO coords back to pixel coords
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        color     = CLASS_COLORS.get(cls_id, (128, 128, 128))
        cls_name  = class_names.get(cls_id, str(cls_id))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, cls_name, (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )

    # Save to verify directory
    verify_dir = Path("data/verify")
    verify_dir.mkdir(exist_ok=True)
    out_path = verify_dir / img_path.name
    cv2.imwrite(str(out_path), img)


def main():
    load_env()

    dataset_config = load_config("configs/dataset.yaml")
    class_names    = {
        idx: name
        for idx, name in dataset_config["classes"].items()
    }

    # Gather all training images that have a corresponding label
    images_dir = Path("data/yolo_dataset/images/train")
    labels_dir = Path("data/yolo_dataset/labels/train")

    all_images = sorted(images_dir.glob("*.png"))
    paired = [
        img for img in all_images
        if (labels_dir / (img.stem + ".txt")).exists()
    ]

    if not paired:
        logger.error("No image+label pairs found. Run convert_annotations.py first.")
        sys.exit(1)

    random.seed(RANDOM_SEED)
    sample = random.sample(paired, min(VERIFY_SAMPLES, len(paired)))

    logger.info(f"Verifying {len(sample)} random training images...")

    for img_path in sample:
        label_path = labels_dir / (img_path.stem + ".txt")
        draw_annotations(img_path, label_path, class_names)

    logger.info(f"Verification images saved to: data/verify/")
    logger.info("Open those images and confirm:")
    logger.info("  - Green boxes around door openings")
    logger.info("  - Blue boxes around windows")
    logger.info("  - Red boxes around wall segments")
    logger.info("  - Yellow boxes around staircases")
    logger.info("  - Magenta boxes around toilets")
    logger.info("  - Orange boxes around sinks")
    logger.info("  - Boxes are correctly sized and positioned (not wildly off)")
    logger.info("  - No boxes covering the wrong elements")


if __name__ == "__main__":
    main()
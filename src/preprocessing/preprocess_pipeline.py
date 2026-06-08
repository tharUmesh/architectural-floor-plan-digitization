"""
Preprocessing Pipeline — Phase 3

PURPOSE:
    Transforms CubiCasa5k floor plan PNGs into model-ready images at 1280×1280.

    CubiCasa5k images are digitally rendered (not scanned), so they arrive
    clean: no uneven illumination, no ink bleed, no paper degradation, and no
    scan noise. The Phase 2 pipeline was designed for scanned blueprints and
    applied several operations (bilateral denoising, CLAHE) that are either
    redundant or actively harmful on already-clean digital renders. This
    version corrects that mismatch.

    Core changes vs Phase 2:
      • Removed bilateral filter — denoising clean images softens the precise
        thin lines that distinguish wall corners, window slots, etc.
      • Removed Otsu's binarization and morphological closing — these were
        never in Phase 2 code but were in the original proposal; confirmed
        absent and intentionally kept out.
      • CLAHE is now optional (APPLY_CLAHE flag). For most CubiCasa5k subsets
        it provides no measurable benefit; test empirically against baseline.
      • Target resolution upgraded 1024→1280. YOLOv11 supports imgsz=1280
        natively via the training parameter, and the extra resolution directly
        helps detect small elements (wall corners, window slots, sinks).
      • Images are loaded with cv2.IMREAD_COLOR (BGR, 3-channel), which is
        what YOLO expects. No grayscale conversion.
      • All remaining internal normalization (mean subtraction, [0,1] scaling)
        is handled by YOLO's own data pipeline; we do not replicate it here.

AUGMENTATION NOTE (applied in YOLO training config, not here):
    Recommended additions to your ultralytics YAML / train() call:
        fliplr=0.5        # horizontal flip — valid for floor plans
        flipud=0.5        # vertical flip   — valid for floor plans
        hsv_h=0.015       # hue jitter      — handles palette diversity
        hsv_s=0.4         # saturation jitter
        hsv_v=0.2         # value jitter
        copy_paste=0.3    # rare-class oversampling (toilets, sinks, corners)
        degrees=10        # small-angle rotation for lightly skewed scans
        mosaic=1.0        # keep existing mosaic augmentation

HOW TO RUN:
    python src/preprocessing/preprocess_pipeline.py

    To test CLAHE variant on a small held-out set, set APPLY_CLAHE = True
    and run a separate experiment. Compare mAP@50-95 against the default
    (APPLY_CLAHE = False) before committing to either approach.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_SIZE = 1280          # YOLOv11 native; directly improves small-element recall
PAD_COLOR   = (114, 114, 114)  # YOLO standard letterbox gray

# Optional CLAHE — disabled by default for clean CubiCasa5k renders.
# Enable only after confirming it improves mAP on a held-out validation set.
APPLY_CLAHE = False
CLAHE_CLIP_LIMIT  = 2.0
CLAHE_TILE_GRID   = (8, 8)


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------

def apply_clahe_lab(img_bgr: np.ndarray) -> np.ndarray:
    """
    Optional CLAHE on the luminance channel only (LAB color space).

    Enhances local contrast without altering hue or saturation and without
    amplifying noise. Only call this function when APPLY_CLAHE is True and
    only after validating it improves mAP on your specific subset.

    Why LAB and not direct grayscale CLAHE:
      CubiCasa5k "colorful" images encode semantic meaning in color (walls,
      doors, and windows use distinct hues). Applying CLAHE directly to a
      grayscale conversion destroys that signal. Operating on the L channel
      in LAB space preserves the full color information while still boosting
      edge contrast.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_enhanced = clahe.apply(l_ch)

    lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def letterbox_image_and_labels(
    img: np.ndarray,
    labels: list[str],
    target_size: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Resize image to target_size while preserving aspect ratio, pad to square
    with YOLO standard gray, and recalculate YOLO bounding box coordinates.

    Letterboxing (rather than squash-resize) is essential for architectural
    drawings, which routinely have extreme aspect ratios. Squash-resizing
    would distort door arcs and wall-corner angles, corrupting the geometric
    features the model needs to learn.

    Bounding box math:
        1. Denormalize YOLO coords to original pixel space.
        2. Apply the same uniform scale factor used to resize the image.
        3. Shift by the symmetric padding offsets.
        4. Renormalize against the new square canvas (target_size × target_size).
    """
    h, w = img.shape[:2]

    # --- resize ---
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # --- symmetric padding ---
    pad_w     = target_size - new_w
    pad_h     = target_size - new_h
    pad_left  = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top   = pad_h // 2
    pad_bot   = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bot, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=PAD_COLOR,
    )

    # --- bounding box adjustment ---
    new_labels: list[str] = []
    for line in labels:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # skip malformed annotations silently

        cls_id = int(parts[0])
        cx_n, cy_n, bw_n, bh_n = map(float, parts[1:])

        # Denormalize → pixel coords in the original image
        cx_px = cx_n * w
        cy_px = cy_n * h
        bw_px = bw_n * w
        bh_px = bh_n * h

        # Scale + shift into the padded canvas
        cx_new = cx_px * scale + pad_left
        cy_new = cy_px * scale + pad_top
        bw_new = bw_px * scale
        bh_new = bh_px * scale

        # Renormalize to [0, 1] on the padded canvas
        cx_f = cx_new / target_size
        cy_f = cy_new / target_size
        bw_f = bw_new / target_size
        bh_f = bh_new / target_size

        # Guard against floating-point drift at the canvas boundary
        cx_f = min(max(cx_f, 0.0), 1.0)
        cy_f = min(max(cy_f, 0.0), 1.0)

        new_labels.append(
            f"{cls_id} {cx_f:.6f} {cy_f:.6f} {bw_f:.6f} {bh_f:.6f}"
        )

    return padded, new_labels


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_dataset() -> None:
    """
    Iterate over train / val / test splits, apply the preprocessing pipeline
    to each image, and write outputs to data/yolo_dataset_processed/.

    Input layout (YOLO format, already split):
        data/yolo_dataset/
            images/{train,val,test}/*.png
            labels/{train,val,test}/*.txt

    Output layout (identical structure, processed):
        data/yolo_dataset_processed/
            images/{train,val,test}/*.png
            labels/{train,val,test}/*.txt
    """
    splits   = ["train", "val", "test"]
    base_dir = Path("data/yolo_dataset")
    out_dir  = Path("data/yolo_dataset_processed")

    clahe_note = "with CLAHE" if APPLY_CLAHE else "no CLAHE (baseline)"
    logger.info(
        f"Pipeline config — target: {TARGET_SIZE}×{TARGET_SIZE}, {clahe_note}"
    )

    for split in splits:
        img_dir   = base_dir / "images" / split
        label_dir = base_dir / "labels" / split

        out_img_dir   = out_dir / "images" / split
        out_label_dir = out_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(img_dir.glob("*.png"))
        if not images:
            logger.warning(f"No PNG images found in {img_dir}, skipping.")
            continue

        logger.info(f"  Processing {split} split — {len(images)} images …")

        for img_path in tqdm(images, desc=split):
            # ---- load ----
            # cv2.IMREAD_COLOR: always returns a 3-channel BGR image.
            # CubiCasa5k PNGs may have an alpha channel; this drops it cleanly.
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"    Could not read {img_path}, skipping.")
                continue

            # ---- load labels ----
            label_path = label_dir / (img_path.stem + ".txt")
            labels: list[str] = []
            if label_path.exists():
                with open(label_path) as f:
                    labels = f.readlines()

            # ---- optional CLAHE ----
            if APPLY_CLAHE:
                img = apply_clahe_lab(img)

            # ---- letterbox + coordinate adjustment ----
            final_img, final_labels = letterbox_image_and_labels(
                img, labels, TARGET_SIZE
            )

            # ---- save ----
            cv2.imwrite(str(out_img_dir / img_path.name), final_img)

            if final_labels:
                with open(out_label_dir / label_path.name, "w") as f:
                    f.write("\n".join(final_labels))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting preprocessing pipeline (Phase 3).")
    process_dataset()
    logger.info("Done. Processed dataset written to data/yolo_dataset_processed/.")

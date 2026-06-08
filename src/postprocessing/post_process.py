"""
Post-Processing & Spatial Mapping Script — Stage 4

PURPOSE:
    Takes raw YOLO predictions, applies NMS/Confidence filters,
    runs geometric sanity checks, and calculates real-world scale (PPM).

HOW TO RUN:
    python src/models/post_process.py
"""

import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

# Architectural Constants
STANDARD_DOOR_WIDTH_M = 0.9  # Standard residential door is ~900mm
DOOR_CLASS_ID = 0  # Ensure this matches your dataset.yaml
WALL_CLASS_ID = 2


def calculate_ppm(detections) -> float:
    """
    Calculates Pixels-Per-Meter (PPM) based on detected doors.
    Accounts for both horizontal and vertical door orientations.
    """
    door_lengths = []
    for det in detections:
        if det["class_id"] == DOOR_CLASS_ID:
            # The 0.9m opening will always be the longest dimension of the bounding box
            actual_opening_px = max(det["width_px"], det["height_px"])
            door_lengths.append(actual_opening_px)

    if not door_lengths:
        print("[WARNING] No doors detected. Spatial mapping will be uncalibrated.")
        return 100.0  # Safe fallback

    # Average the length of all detected doors to smooth out pixel jitter
    avg_door_px = sum(door_lengths) / len(door_lengths)
    ppm = avg_door_px / STANDARD_DOOR_WIDTH_M
    return ppm


def geometric_sanity_check(det) -> bool:
    """
    Filters out mathematically impossible architectural elements.
    Returns True if the detection makes sense, False if it should be discarded.
    """
    w, h = det["width_px"], det["height_px"]
    aspect_ratio = max(w, h) / min(w, h)

    # Example check: Walls shouldn't be perfect squares
    if det["class_id"] == WALL_CLASS_ID and aspect_ratio < 1.5:
        return False

    return True


def process_floorplan(
    image_path: str, model_path: str, conf_thresh=0.25, iou_thresh=0.45
):
    """
    Runs the full Stage 4 pipeline on a single image.
    """
    print(f"Processing: {Path(image_path).name}")

    # 1. Load Model & Run Inference (NMS and Conf filtering happen here)
    model = YOLO(model_path)
    results = model.predict(
        source=image_path, conf=conf_thresh, iou=iou_thresh, verbose=False
    )[0]

    # Extract raw bounding boxes
    boxes = results.boxes
    names = model.names

    raw_detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()

        raw_detections.append(
            {
                "class_id": cls_id,
                "class_name": names[cls_id],
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width_px": x2 - x1,
                "height_px": y2 - y1,
            }
        )

    # 2. Class-Level Validation (Geometric Sanity Checks)
    filtered_detections = [d for d in raw_detections if geometric_sanity_check(d)]

    # 3. Spatial Mapping (Calculate PPM)
    ppm = calculate_ppm(filtered_detections)
    print(f"Calculated Scale: {ppm:.2f} Pixels Per Meter")

    # 4. Attach Real-World Coordinates
    final_elements = []
    for det in filtered_detections:
        det["width_m"] = det["width_px"] / ppm
        det["height_m"] = det["height_px"] / ppm
        final_elements.append(det)

    print(
        f"Retained {len(final_elements)} valid architectural elements after filtering."
    )
    return final_elements, ppm


if __name__ == "__main__":
    # Test the pipeline on a single image
    TEST_IMAGE = "data/yolo_dataset_processed/images/test/colorful_10711_F1.png"
    MODEL_WEIGHTS = "models/yolo11l_step2/weights/best.pt"

    if Path(TEST_IMAGE).exists():
        elements, scale = process_floorplan(TEST_IMAGE, MODEL_WEIGHTS)

        # Print a sample of the cleaned data
        if elements:
            print("\nSample Extracted Element:")
            print(elements[0])
    else:
        print("Please update the TEST_IMAGE path to a valid test image.")

"""
Vectorization & Export Script — Stage 5

PURPOSE:
    Converts post-processed YOLO detections into structured JSON and
    AutoCAD-compatible DXF files.  Integrates directly with the
    post_process.py pipeline so that a single command runs inference
    on a floor plan image and exports a fully viewable CAD file.

FIXES (from blank-DXF issue):
    1. The old __main__ used hardcoded sample data → now runs real inference.
    2. $EXTMIN/$EXTMAX were unset → now explicitly computed and written
       so that CAD viewers auto-zoom to the drawing extents.
    3. Added text labels for each element in the DXF.
    4. Added all 6 class layers (was missing Staircase/Toilet/Sink layers).
    5. Added INSUNITS=6 (meters) in the header so CAD reads correct scale.

HOW TO RUN (standalone full pipeline):
    python src/export/export_cad.py                     # Processes default test image
    python src/export/export_cad.py <image_path>        # Processes any image
"""

import json
import sys
from pathlib import Path

import cv2
import ezdxf

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.postprocessing.post_process import process_floorplan

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_WEIGHTS = "runs/detect/models/yolo11l_step2-2/weights/best.pt"
OUTPUT_DIR = Path("runs/exports")
DEFAULT_TEST_IMAGE = "data/yolo_dataset_processed/images/test/colorful_10711_F1.png"

# Layer configuration: class_name → (layer_name, DXF color index)
LAYER_CONFIG = {
    "Door":      ("DOORS",      3),   # Green
    "Window":    ("WINDOWS",    5),   # Blue
    "Wall":      ("WALLS",      1),   # Red
    "Staircase": ("STAIRCASES", 4),   # Cyan
    "Toilet":    ("TOILETS",    6),   # Magenta
    "Sink":      ("SINKS",      2),   # Yellow
}


def export_to_json(elements: list, output_path: str):
    """Saves the extracted architectural elements to a structured JSON file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        json.dump({"architectural_elements": elements}, f, indent=4)

    print(f"  JSON export: {out_file}")


def export_to_dxf(elements: list, output_path: str, img_height_px: float, ppm: float):
    """
    Generates a DXF CAD file from detected architectural elements.

    Key fixes:
      - Sets $EXTMIN/$EXTMAX so CAD viewers auto-zoom to content.
      - Sets $INSUNITS=6 (meters) for correct scaling.
      - Adds text labels next to each element for readability.
      - Supports all 6 architectural classes with distinct layers/colors.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a new DXF document (R2010 = AutoCAD 2010 format, widely supported)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Setup CAD Layers — one per class with distinct colors
    for class_name, (layer_name, color) in LAYER_CONFIG.items():
        doc.layers.add(layer_name, color=color)
    doc.layers.add("LABELS", color=7)  # White — for text annotations

    # Track drawing extents for viewport fix
    all_x, all_y = [], []

    for det in elements:
        # 1. Determine correct layer
        class_name = det["class_name"]
        layer_name, _ = LAYER_CONFIG.get(class_name, ("FIXTURES", 6))

        # 2. Invert Y-axis (image top-left → CAD bottom-left origin)
        cad_y1_px = img_height_px - det["y2"]
        cad_y2_px = img_height_px - det["y1"]

        # 3. Convert pixel coordinates to real-world meters
        x1_m = det["x1"] / ppm
        y1_m = cad_y1_px / ppm
        x2_m = det["x2"] / ppm
        y2_m = cad_y2_px / ppm

        # Track extents
        all_x.extend([x1_m, x2_m])
        all_y.extend([y1_m, y2_m])

        # 4. Draw the bounding box as a closed polyline
        points = [
            (x1_m, y1_m),
            (x2_m, y1_m),
            (x2_m, y2_m),
            (x1_m, y2_m),
            (x1_m, y1_m),  # Close the loop
        ]
        msp.add_lwpolyline(points, dxfattribs={"layer": layer_name})

        # 5. Add a text label at the top-left corner of each element
        label = f"{class_name} ({det['confidence']:.0%})"
        text_height = max(0.08, min((y2_m - y1_m) * 0.3, 0.3))
        msp.add_text(
            label,
            dxfattribs={
                "layer": "LABELS",
                "height": text_height,
                "insert": (x1_m, y2_m + 0.05),
            },
        )

    # ── Fix viewport extents ──────────────────────────────────────────────
    # This is the KEY fix: without valid $EXTMIN/$EXTMAX, CAD viewers
    # don't know where the drawing content is, so they show a blank canvas.
    if all_x and all_y:
        margin = 0.5  # Add 0.5m margin around the drawing
        min_x, max_x = min(all_x) - margin, max(all_x) + margin
        min_y, max_y = min(all_y) - margin, max(all_y) + margin

        doc.header["$EXTMIN"] = (min_x, min_y, 0)
        doc.header["$EXTMAX"] = (max_x, max_y, 0)
        doc.header["$LIMMIN"] = (min_x, min_y)
        doc.header["$LIMMAX"] = (max_x, max_y)

    # Set drawing units to meters
    doc.header["$INSUNITS"] = 6  # 6 = meters in DXF standard

    # Save the document
    doc.saveas(str(out_file))
    print(f"  DXF export:  {out_file}")
    if all_x:
        print(f"  Drawing extents: ({min(all_x):.1f}, {min(all_y):.1f}) to ({max(all_x):.1f}, {max(all_y):.1f}) meters")
        print(f"  Total elements: {len(elements)}")


def run_full_pipeline(image_path: str, model_path: str = MODEL_WEIGHTS):
    """
    End-to-end pipeline: Image → YOLO Inference → Post-Processing → CAD Export

    This is the function that was MISSING — the old code had post_process.py
    and export_cad.py as disconnected scripts.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    print(f"\n{'='*60}")
    print(f"  FLOOR PLAN → CAD PIPELINE")
    print(f"  Image: {image_path.name}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")

    # Step 1: Run inference + post-processing
    print("\n[1/3] Running YOLO inference + post-processing...")
    elements, ppm = process_floorplan(str(image_path), model_path)

    if not elements:
        print("WARNING: No architectural elements detected! CAD file will be empty.")
        print("Check that the model path is correct and the image is a valid floor plan.")
        return

    # Step 2: Get image dimensions for Y-axis inversion
    img = cv2.imread(str(image_path))
    img_height_px = img.shape[0]

    # Step 3: Export
    stem = image_path.stem
    output_dir = OUTPUT_DIR / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/3] Exporting to JSON...")
    export_to_json(elements, str(output_dir / f"{stem}.json"))

    print(f"[3/3] Exporting to DXF...")
    export_to_dxf(elements, str(output_dir / f"{stem}.dxf"), img_height_px, ppm)

    print(f"\n{'='*60}")
    print(f"  EXPORT COMPLETE")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")

    return elements


if __name__ == "__main__":
    # Accept optional image path from command line
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    else:
        input_image = DEFAULT_TEST_IMAGE

    run_full_pipeline(input_image)

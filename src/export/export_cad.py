"""
Vectorization & Export Script — Stage 5

PURPOSE:
    Converts real-world scaled bounding boxes into structured JSON
    and AutoCAD-compatible DXF files.

HOW TO RUN:
    This is designed to be imported into your main pipeline, but can
    be tested directly via: python src/models/export_cad.py
"""

import json
from pathlib import Path
import ezdxf


def export_to_json(elements: list, output_path: str):
    """
    Saves the extracted architectural elements to a structured JSON file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        json.dump({"architectural_elements": elements}, f, indent=4)

    print(f"JSON export successful: {out_file.name}")


def export_to_dxf(elements: list, output_path: str, img_height_px: float, ppm: float):
    """
    Generates a DXF CAD file. Converts pixel coordinates to meters and
    inverts the Y-axis to match standard CAD coordinate systems.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a new DXF document
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Setup CAD Layers with standard colors
    doc.layers.add("WALLS", color=1)  # Red
    doc.layers.add("DOORS", color=3)  # Green
    doc.layers.add("WINDOWS", color=5)  # Blue
    doc.layers.add("FIXTURES", color=6)  # Magenta (Sinks/Toilets)

    for det in elements:
        # 1. Determine correct layer
        layer_name = "FIXTURES"
        if det["class_name"] == "Wall":
            layer_name = "WALLS"
        elif det["class_name"] == "Door":
            layer_name = "DOORS"
        elif det["class_name"] == "Window":
            layer_name = "WINDOWS"

        # 2. Invert Y-Axis (Image top-left to CAD bottom-left)
        # We subtract the y-coordinates from the total image height
        cad_y1_px = img_height_px - det["y2"]
        cad_y2_px = img_height_px - det["y1"]

        # 3. Convert to Real-World Meters
        x1_m = det["x1"] / ppm
        y1_m = cad_y1_px / ppm
        x2_m = det["x2"] / ppm
        y2_m = cad_y2_px / ppm

        # 4. Draw the bounding box in the DXF Modelspace
        # A rectangle is formed by 4 lines connecting the corners
        points = [
            (x1_m, y1_m),
            (x2_m, y1_m),
            (x2_m, y2_m),
            (x1_m, y2_m),
            (x1_m, y1_m),  # Close the loop
        ]

        # Draw a polyline representing the element
        msp.add_lwpolyline(points, dxfattribs={"layer": layer_name})

    # Save the document
    doc.saveas(str(out_file))
    print(f"DXF export successful: {out_file.name}")


if __name__ == "__main__":
    # --- Quick Integration Test ---
    # This mocks the data you just received from post_process.py
    sample_data = [
        {
            "class_id": 1,
            "class_name": "Window",
            "confidence": 0.693,
            "x1": 509.0,
            "y1": 877.4,
            "x2": 592.1,
            "y2": 914.6,
            "width_px": 83.1,
            "height_px": 37.2,
            "width_m": 1.03,
            "height_m": 0.46,
        }
    ]

    # Assuming standard 1024x1024 image size from our pipeline and your PPM
    export_to_json(sample_data, "runs/exports/colorful_10711_F1.json")
    export_to_dxf(
        sample_data, "runs/exports/colorful_10711_F1.dxf", img_height_px=1024, ppm=80.67
    )

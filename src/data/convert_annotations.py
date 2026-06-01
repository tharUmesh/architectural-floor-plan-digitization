"""
Annotation Conversion Script — Phase 2

PURPOSE:
    Convert CubiCasa5k SVG annotations → YOLO .txt label files.
    Copy corresponding PNG images to the YOLO dataset directory.

    Two extraction patterns (discovered in Phase 1 SVG audit):
      Pattern A — Structural (Wall, Door, Window, Stairs):
        Direct polygon coordinates. First direct-child <polygon> = boundary.

      Pattern B — Furniture (Toilet, Sink):
        Local polygon coordinates inside a BoundaryPolygon child.
        Parent <g> carries transform="matrix(a,b,c,d,e,f)".
        Must apply matrix transform to get real image coordinates.

OUTPUT:
    data/yolo_dataset/images/{train,val,test}/   ← PNG images
    data/yolo_dataset/labels/{train,val,test}/   ← YOLO .txt labels

    YOLO label format (one line per object):
    class_id  x_center  y_center  width  height
    (all values normalized to [0.0, 1.0] relative to image dimensions)

HOW TO RUN:
    make convert
    OR
    python src/data/convert_annotations.py
"""

import json
import re
import shutil
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config, load_env, get_env
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Element classification ────────────────────────────────────────────────────
# Which keywords use Pattern A (direct coordinates)
STRUCTURAL_KEYWORDS = {"Door", "Window", "Wall"}
# Which keywords use Pattern B (local coords + matrix transform)
FURNITURE_KEYWORDS  = {"Toilet", "Sink"}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def strip_ns(tag: str) -> str:
    """Remove XML namespace prefix. '{http://...}polygon' → 'polygon'"""
    return tag.split("}")[-1] if "}" in tag else tag


def parse_points(points_str: str) -> list[tuple[float, float]]:
    """
    Parse an SVG polygon 'points' attribute into a list of (x, y) tuples.

    CubiCasa format: "x1,y1 x2,y2 x3,y3 ..."
    Handles edge cases where commas and spaces are mixed inconsistently.

    Returns empty list if parsing fails (caller handles this gracefully).
    """
    try:
        # Normalize: replace all commas with spaces, then split on whitespace
        tokens = points_str.replace(",", " ").split()
        nums = [float(t) for t in tokens if t]

        # Must have at least 3 coordinate pairs (triangle) to form a real shape
        if len(nums) < 6 or len(nums) % 2 != 0:
            return []

        return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
    except (ValueError, IndexError):
        return []


def coords_to_bbox(
    coords: list[tuple[float, float]]
) -> tuple[float, float, float, float] | None:
    """
    Compute axis-aligned bounding box from a list of (x, y) points.

    Returns (x_min, y_min, x_max, y_max) or None if coords is empty.
    """
    if not coords:
        return None
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return min(xs), min(ys), max(xs), max(ys)


def parse_matrix_transform(transform_str: str) -> tuple | None:
    """
    Parse an SVG transform="matrix(a,b,c,d,e,f)" attribute.

    The 6 values define an affine transformation:
      real_x = a*local_x + c*local_y + e
      real_y = b*local_x + d*local_y + f

    Returns (a, b, c, d, e, f) as floats, or None if parsing fails.
    """
    if not transform_str:
        return None
    match = re.search(r"matrix\(([^)]+)\)", transform_str)
    if not match:
        return None
    try:
        values = [float(v) for v in match.group(1).replace(",", " ").split()]
        if len(values) != 6:
            return None
        return tuple(values)
    except ValueError:
        return None


def apply_matrix(
    coords: list[tuple[float, float]],
    matrix: tuple
) -> list[tuple[float, float]]:
    """
    Apply a 2D affine matrix transform to a list of (x, y) points.

    matrix = (a, b, c, d, e, f)
    real_x = a*x + c*y + e
    real_y = b*x + d*y + f
    """
    a, b, c, d, e, f = matrix
    return [(a * x + c * y + e, b * x + d * y + f) for x, y in coords]

def compose_matrices(
    parent_m: tuple | None,
    child_m:  tuple | None,
) -> tuple | None:
    """
    Compose two affine matrix transforms into one.

    SVG transforms are cumulative: a child element's local coordinates
    are transformed by its own matrix first, then its parent's matrix.
    Composing them produces a single matrix that does both in one step.

    Math (standard 2D affine matrix composition):
      Given parent=(a1,b1,c1,d1,e1,f1) and child=(a2,b2,c2,d2,e2,f2):
      combined = (
          a2*a1 + c2*b1,
          b2*a1 + d2*b1,
          a2*c1 + c2*d1,
          b2*c1 + d2*d1,
          a2*e1 + c2*f1 + e2,
          b2*e1 + d2*f1 + f2,
      )
    """
    if parent_m is None and child_m is None:
        return None
    if parent_m is None:
        return child_m
    if child_m is None:
        return parent_m

    a1, b1, c1, d1, e1, f1 = parent_m
    a2, b2, c2, d2, e2, f2 = child_m

    return (
        a2 * a1 + c2 * b1,
        b2 * a1 + d2 * b1,
        a2 * c1 + c2 * d1,
        b2 * c1 + d2 * d1,
        a2 * e1 + c2 * f1 + e2,
        b2 * e1 + d2 * f1 + f2,
    )

def bbox_to_yolo(
    bbox: tuple[float, float, float, float],
    svg_w: float,
    svg_h: float,
) -> tuple[float, float, float, float] | None:
    """
    Convert a pixel bounding box to YOLO normalized format.

    YOLO format: (x_center, y_center, width, height) all in [0.0, 1.0]
    Normalization is by SVG viewBox dimensions (equivalent to normalizing
    by PNG pixel dimensions since F1_scaled.png renders at SVG dimensions).

    Returns None if the resulting box is invalid (zero area, out of bounds).
    """
    x_min, y_min, x_max, y_max = bbox

    # Clamp to SVG bounds (handles tiny floating-point overflows at edges)
    x_min = max(0.0, min(x_min, svg_w))
    y_min = max(0.0, min(y_min, svg_h))
    x_max = max(0.0, min(x_max, svg_w))
    y_max = max(0.0, min(y_max, svg_h))

    box_w = x_max - x_min
    box_h = y_max - y_min

    # Reject zero-area and extremely tiny boxes
    # (less than 0.1% of image dimension in either axis)
    if box_w < svg_w * 0.001 or box_h < svg_h * 0.001:
        return None

    x_center = (x_min + x_max) / 2.0 / svg_w
    y_center = (y_min + y_max) / 2.0 / svg_h
    w_norm   = box_w / svg_w
    h_norm   = box_h / svg_h

    return x_center, y_center, w_norm, h_norm


# ─────────────────────────────────────────────────────────────────────────────
# Bounding box extraction — Pattern A and Pattern B
# ─────────────────────────────────────────────────────────────────────────────

def extract_structural_bbox(
    elem: ET.Element,
    accumulated_transform: tuple | None = None,
) -> tuple | None:
    """
    Pattern A: Bounding box from first direct-child <polygon>.
    Used for: Wall, Door, Window.

    These elements' polygon points are already in SVG canvas coordinates.
    accumulated_transform is applied as a safety measure (should be None
    for structural elements in CubiCasa5k, making it a no-op).
    """
    for child in elem:
        if strip_ns(child.tag) == "polygon":
            coords = parse_points(child.get("points", ""))
            if coords:
                if accumulated_transform:
                    coords = apply_matrix(coords, accumulated_transform)
                return coords_to_bbox(coords)
    return None


def extract_stairs_bbox(
    elem: ET.Element,
    accumulated_transform: tuple | None = None,
) -> tuple | None:
    """
    Stairs: collect polygon points from ALL descendants and compute
    the union bounding box.

    Stairs elements have NO direct polygon child. All geometry lives
    inside nested <g class="Steps"> children:

        <g class="Stairs">
            <g class="Steps">
                <polygon points="..."/>   ← one step line
                <polygon points="..."/>   ← another step line
            </g>
            <g class="Steps">
                <polygon points="..."/>
            </g>
        </g>

    The bounding box of ALL step polygons together = the staircase area.
    """
    all_coords = []

    for descendant in elem.iter():
        if strip_ns(descendant.tag) != "polygon":
            continue
        pts = parse_points(descendant.get("points", ""))
        if pts:
            if accumulated_transform:
                pts = apply_matrix(pts, accumulated_transform)
            all_coords.extend(pts)

    return coords_to_bbox(all_coords) if all_coords else None


def extract_furniture_bbox(
    elem: ET.Element,
    accumulated_transform: tuple | None = None,
) -> tuple | None:
    """
    Pattern B: Bounding box from BoundaryPolygon child, with accumulated
    transform applied.
    Used for: Toilet, Sink.

    The accumulated_transform is the composition of ALL ancestor transforms
    down to this element. This correctly handles both:
      - Transform directly on the furniture element (Toilet case)
      - Transform on a parent wrapper group (Sink-in-FixedFurnitureSet case)

    BoundaryPolygon points are in the element's local coordinate space.
    Applying the accumulated transform converts them to canvas coordinates.
    """
    for child in elem:
        if strip_ns(child.tag) != "g":
            continue
        if "BoundaryPolygon" not in child.get("class", "").split():
            continue

        for grandchild in child:
            if strip_ns(grandchild.tag) == "polygon":
                coords = parse_points(grandchild.get("points", ""))
                if coords:
                    if accumulated_transform:
                        coords = apply_matrix(coords, accumulated_transform)
                    return coords_to_bbox(coords)

    # Fallback: if no BoundaryPolygon found, try structural pattern
    return extract_structural_bbox(elem, accumulated_transform)


# ─────────────────────────────────────────────────────────────────────────────
# SVG annotation extractor
# ─────────────────────────────────────────────────────────────────────────────

def get_svg_dimensions(root: ET.Element) -> tuple[float, float] | None:
    """
    Extract the SVG canvas dimensions from the root <svg> element.

    Tries 'width'/'height' attributes first.
    Falls back to parsing the 'viewBox' attribute.

    These dimensions define the coordinate space for all polygon points,
    and are used as the normalization denominator for YOLO format.
    """
    try:
        width  = float(root.get("width",  0))
        height = float(root.get("height", 0))
        if width > 0 and height > 0:
            return width, height
    except (ValueError, TypeError):
        pass

    # Fallback: parse viewBox="min_x min_y width height"
    viewbox = root.get("viewBox", "")
    if viewbox:
        try:
            parts = viewbox.replace(",", " ").split()
            return float(parts[2]), float(parts[3])
        except (ValueError, IndexError):
            pass

    return None


def extract_annotations(
    svg_path: Path,
    class_id_map: dict[str, int],
    svg_class_tags: dict[str, list[str]],
) -> tuple[list[str], dict]:
    """
    Parse one SVG file and extract all YOLO-format annotations.

    Uses recursive depth-first traversal to accumulate transforms correctly.
    Each element is processed with the full transform chain from root to
    that element, ensuring furniture positioned by ancestor transforms is
    correctly located.
    """
    yolo_lines = []
    stats = {
        "found":   {cls: 0 for cls in class_id_map},
        "skipped": {cls: 0 for cls in class_id_map},
    }

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        logger.error(f"SVG parse error in {svg_path}: {exc}")
        return yolo_lines, stats

    svg_dims = get_svg_dimensions(root)
    if svg_dims is None:
        logger.warning(f"Cannot determine SVG dimensions: {svg_path}")
        return yolo_lines, stats
    svg_w, svg_h = svg_dims

    # Reverse lookup: SVG keyword → (class_name, class_id)
    keyword_to_class: dict[str, tuple[str, int]] = {}
    for class_name, keywords in svg_class_tags.items():
        if class_name in class_id_map:
            for kw in keywords:
                keyword_to_class[kw] = (class_name, class_id_map[class_name])

    # ── Recursive traversal tracking accumulated transform ────────────────
    def traverse(elem: ET.Element, accumulated: tuple | None) -> None:
        """
        Visit elem with its accumulated ancestor transform.
        Computes the new accumulated transform for this element,
        checks if it matches a target class, extracts bbox if so,
        then recurses into children.
        """
        # Skip non-group elements at the top level
        # (we only match <g> elements, but we must still recurse
        #  through non-g elements to reach nested <g> children)
        tag = strip_ns(elem.tag)

        # Accumulate this element's own transform
        my_transform  = parse_matrix_transform(elem.get("transform", ""))
        new_accumulated = compose_matrices(accumulated, my_transform)

        if tag == "g":
            class_str   = elem.get("class", "")
            id_str      = elem.get("id",    "")
            class_words = set(class_str.split())

            # Check if this element matches any target class
            matched_class_name = None
            matched_class_id   = None
            matched_keyword    = None

            for keyword, (cls_name, cls_id) in keyword_to_class.items():
                if keyword in class_words or id_str == keyword:
                    matched_class_name = cls_name
                    matched_class_id   = cls_id
                    matched_keyword    = keyword
                    break

            if matched_class_name is not None:
                # ── Extract bounding box ──────────────────────────────────
                bbox = None

                if matched_keyword in STRUCTURAL_KEYWORDS:
                    bbox = extract_structural_bbox(elem, new_accumulated)

                elif matched_keyword in FURNITURE_KEYWORDS:
                    bbox = extract_furniture_bbox(elem, new_accumulated)

                else:
                    # Stairs: collect all descendant polygon points
                    bbox = extract_stairs_bbox(elem, new_accumulated)

                if bbox is None:
                    stats["skipped"][matched_class_name] += 1
                else:
                    yolo_coords = bbox_to_yolo(bbox, svg_w, svg_h)
                    if yolo_coords is None:
                        stats["skipped"][matched_class_name] += 1
                    else:
                        x_c, y_c, w, h = yolo_coords
                        yolo_lines.append(
                            f"{matched_class_id} "
                            f"{x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                        )
                        stats["found"][matched_class_name] += 1

        # Always recurse into children regardless of whether this element matched
        for child in elem:
            traverse(child, new_accumulated)

    traverse(root, None)
    return yolo_lines, stats


# ─────────────────────────────────────────────────────────────────────────────
# Main processing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_class_maps(
    dataset_config: dict,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Build class_id_map and svg_class_tags from config."""
    class_id_map   = {name: idx for idx, name in dataset_config["classes"].items()}
    svg_class_tags = dataset_config["svg_class_tags"]
    return class_id_map, svg_class_tags


def process_split(
    split_name: str,
    folder_list: list[str],
    image_filename: str,
    class_id_map: dict[str, int],
    svg_class_tags: dict[str, list[str]],
    yolo_images_dir: Path,
    yolo_labels_dir: Path,
) -> dict:
    """
    Process all folders in one split (train/val/test).

    For each folder:
      1. Build a unique output filename from category + folder name
      2. Copy PNG → yolo_images_dir
      3. Extract annotations from SVG → write .txt to yolo_labels_dir

    Returns aggregate statistics for reporting.
    """
    agg_stats = {
        "total_folders":  0,
        "skipped_no_img": 0,
        "skipped_no_ann": 0,
        "total_found":    {cls: 0 for cls in class_id_map},
        "total_skipped":  {cls: 0 for cls in class_id_map},
    }

    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)

    for folder_str in tqdm(folder_list, desc=f"  {split_name}", unit="folder"):
        folder    = Path(folder_str)
        img_path  = folder / image_filename
        svg_path  = folder / "model.svg"

        # Generate a unique, collision-free filename
        # e.g. colorful/30 → "colorful_30_F1"
        category    = folder.parent.name          # "colorful"
        folder_name = folder.name                 # "30"
        stem        = f"{category}_{folder_name}_F1"

        out_img_path   = yolo_images_dir / f"{stem}.png"
        out_label_path = yolo_labels_dir / f"{stem}.txt"

        # ── Skip if PNG is missing ────────────────────────────────────────
        if not img_path.exists():
            logger.debug(f"Missing image: {img_path}")
            agg_stats["skipped_no_img"] += 1
            continue

        # ── Extract annotations ───────────────────────────────────────────
        yolo_lines, stats = extract_annotations(
            svg_path, class_id_map, svg_class_tags
        )

        # Skip folders with zero valid annotations
        # (keeps YOLO dataset clean — empty label files cause training warnings)
        if not yolo_lines:
            logger.debug(f"No annotations extracted: {folder}")
            agg_stats["skipped_no_ann"] += 1
            continue

        # ── Write label file ──────────────────────────────────────────────
        out_label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        # ── Copy image ────────────────────────────────────────────────────
        shutil.copy2(img_path, out_img_path)

        # ── Accumulate stats ──────────────────────────────────────────────
        agg_stats["total_folders"] += 1
        for cls in class_id_map:
            agg_stats["total_found"][cls]   += stats["found"].get(cls, 0)
            agg_stats["total_skipped"][cls] += stats["skipped"].get(cls, 0)

    return agg_stats


def print_conversion_report(all_stats: dict[str, dict], class_id_map: dict) -> None:
    """Print a formatted post-conversion summary."""
    print("\n" + "=" * 60)
    print("ANNOTATION CONVERSION COMPLETE")
    print("=" * 60)

    total_folders  = sum(s["total_folders"]  for s in all_stats.values())
    total_skipped  = sum(s["skipped_no_ann"] for s in all_stats.values())

    for split, stats in all_stats.items():
        print(f"\n  {split.upper()}: {stats['total_folders']} folders written"
              f" ({stats['skipped_no_ann']} skipped — no annotations)")

    print(f"\n  Total folders with labels: {total_folders}")
    print(f"  Total folders skipped:     {total_skipped}")

    print(f"\n  Annotations per class (train+val+test):")
    for cls in class_id_map:
        found   = sum(s["total_found"].get(cls, 0)   for s in all_stats.values())
        skipped = sum(s["total_skipped"].get(cls, 0) for s in all_stats.values())
        print(f"    {cls:<12} found={found:>6}  skipped={skipped:>4}")

    print("\n  Output directories:")
    print("    data/yolo_dataset/images/{train,val,test}/")
    print("    data/yolo_dataset/labels/{train,val,test}/")
    print("\n  Next step: run src/data/verify_annotations.py")
    print("=" * 60)


def main():
    load_env()

    dataset_config  = load_config("configs/dataset.yaml")
    class_id_map, svg_class_tags = build_class_maps(dataset_config)
    image_filename  = dataset_config["cubicasa"]["image_filename"]

    splits_dir      = Path("data/splits")
    yolo_base       = Path("data/yolo_dataset")

    logger.info("Starting annotation conversion...")
    logger.info(f"Target classes: {list(class_id_map.keys())}")
    logger.info(f"SVG keywords:   {svg_class_tags}")

    all_stats = {}

    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{split_name}.json"
        if not split_file.exists():
            logger.error(f"Split file not found: {split_file}")
            sys.exit(1)

        with open(split_file) as f:
            split_data = json.load(f)

        folder_list = split_data["folders"]
        logger.info(f"Processing {split_name}: {len(folder_list)} folders")

        stats = process_split(
            split_name     = split_name,
            folder_list    = folder_list,
            image_filename = image_filename,
            class_id_map   = class_id_map,
            svg_class_tags = svg_class_tags,
            yolo_images_dir = yolo_base / "images" / split_name,
            yolo_labels_dir = yolo_base / "labels" / split_name,
        )
        all_stats[split_name] = stats

    print_conversion_report(all_stats, class_id_map)


if __name__ == "__main__":
    main()
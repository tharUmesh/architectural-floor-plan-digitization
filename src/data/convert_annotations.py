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

import cv2
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


def get_element_coords(elem: ET.Element) -> list[tuple[float, float]]:
    """
    Extract (x, y) coordinate pairs from any SVG geometry element.

    Handles three element types:
      <polygon>/<polyline> — coordinates in 'points' attribute
      <rect>               — corners computed from x, y, width, height
      <path>               — numbers extracted from 'd' attribute

    Why path extraction works for bounding boxes:
    We extract ALL floating-point numbers from the path 'd' attribute
    and treat consecutive pairs as (x, y). H (horizontal) and V (vertical)
    single-coordinate commands may slightly misalign pairings, but the
    resulting min/max range across all numbers still correctly bounds the
    shape. For a toilet bowl path like "M40,44 S41,70 20,70 C-0.5,70..."
    this gives x:[-0.5, 41], y:[18, 70] — an accurate bounding box.
    """
    tag = strip_ns(elem.tag)

    if tag in ("polygon", "polyline"):
        return parse_points(elem.get("points", ""))

    if tag == "rect":
        try:
            x = float(elem.get("x", 0))
            y = float(elem.get("y", 0))
            w = float(elem.get("width", 0))
            h = float(elem.get("height", 0))
            if w > 0 and h > 0:
                return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        except (ValueError, TypeError):
            pass

    if tag == "path":
        d = elem.get("d", "")
        if d:
            # Remove all SVG command letters, leaving only numbers
            nums_str = re.sub(r"[MmLlHhVvCcSsQqTtAaZz]", " ", d)
            try:
                nums = [
                    float(n)
                    for n in nums_str.replace(",", " ").split()
                    if n
                ]
                # Pair consecutive numbers as (x, y) — good enough for bbox
                return [
                    (nums[i], nums[i + 1])
                    for i in range(0, len(nums) - 1, 2)
                ]
            except (ValueError, IndexError):
                pass

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


# SVG elements whose subtrees contain only rendering definitions,
# never actual spatial geometry. Skipping them prevents arrowhead
# marker polygons (with near-zero local coords) from corrupting bboxes.
_DEFINITION_TAGS = frozenset({"defs", "symbol", "clipPath", "mask", "marker"})

# Classes that represent visual markers, arrows, labels — NOT structural geometry.
# These are excluded from bounding box calculations to prevent direction arrows
# (which have local near-zero coordinates) from expanding bbox to wrong areas.
_NOISE_CLASSES = frozenset({
    "Direction", "Arrow", "Name", "Label",
    "Text", "North", "Dimension", "SelectionControls",
})


def collect_recursive(
    elem: ET.Element,
    base_accumulated: tuple | None,
    skip_classes: frozenset | None = None,
) -> list[tuple[float, float]]:
    all_coords: list[tuple[float, float]] = []

    def descend(current: ET.Element, current_acc: tuple | None) -> None:
        for child in current:
            tag = strip_ns(child.tag)

            # ── NEW: skip SVG definition containers entirely ──────────────
            # <defs>, <marker>, <symbol> etc. contain rendering definitions
            # (arrowheads, patterns, clip regions), never actual geometry.
            # The staircase direction arrow marker lives inside <defs> and
            # has polygon coordinates near (0,0), corrupting the bbox.
            if tag in _DEFINITION_TAGS:
                continue
            # ─────────────────────────────────────────────────────────────

            child_t   = parse_matrix_transform(child.get("transform", ""))
            child_acc = compose_matrices(current_acc, child_t)

            if tag == "g" and skip_classes:
                child_class_words = set(child.get("class", "").split())
                if child_class_words & skip_classes:
                    continue

            coords = get_element_coords(child)
            if coords:
                if child_acc:
                    coords = apply_matrix(coords, child_acc)
                all_coords.extend(coords)

            descend(child, child_acc)

    descend(elem, base_accumulated)
    return all_coords

def bbox_to_yolo(
    bbox: tuple[float, float, float, float],
    svg_w: float,
    svg_h: float,
) -> tuple[float, float, float, float] | None:
    """
    Convert a pixel bounding box to YOLO normalized format.

    Size threshold change: reduced from 0.001 to 0.0001 (10x more permissive).
    Reason: Small fixtures (toilets, sinks) in large professional drawings
    (mean 1600×1400px) can legitimately produce boxes as small as 0.05% of
    image width. The old 0.1% threshold was incorrectly rejecting these.
    The new 0.01% threshold still filters out genuinely degenerate zero-area
    boxes while keeping all real architectural elements.
    """
    x_min, y_min, x_max, y_max = bbox

    # Clamp to SVG bounds
    x_min = max(0.0, min(x_min, svg_w))
    y_min = max(0.0, min(y_min, svg_h))
    x_max = max(0.0, min(x_max, svg_w))
    y_max = max(0.0, min(y_max, svg_h))

    box_w = x_max - x_min
    box_h = y_max - y_min

    # Reject only truly zero-area boxes (degenerate geometry)
    # 0.0001 = 0.01% of image dimension — much more permissive than before
    if box_w < svg_w * 0.0001 or box_h < svg_h * 0.0001:
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
    Stairs: collect all descendant polygon coords, skipping noise groups.

    The _NOISE_CLASSES filter specifically excludes <g class="Direction">
    elements which contain direction arrow polygons with local near-zero
    coordinates. Without this filter, those arrow coords pull the bounding
    box toward the top-left corner of the image.
    """
    all_coords = collect_recursive(
        elem,
        accumulated_transform,
        skip_classes=_NOISE_CLASSES,
    )
    return coords_to_bbox(all_coords) if all_coords else None

def find_boundary_polygon(
    elem: ET.Element,
    base_accumulated: tuple | None,
) -> tuple | None:
    """
    Recursively search all descendants of elem for class="BoundaryPolygon".
    Collects coordinates from ALL geometry children (polygon, rect, path).

    Why all geometry types: CubiCasa5k furniture elements use different
    geometry inside BoundaryPolygon depending on the fixture type:
      - Simple fixtures (sink, square toilet): <polygon>
      - Complex fixtures (rounded toilet, bathtub): <rect> + <path>
    Supporting only <polygon> caused ~164 toilets to be skipped.
    """
    def search(
        current: ET.Element,
        current_acc: tuple | None,
    ) -> tuple | None:

        for child in current:
            if strip_ns(child.tag) != "g":
                continue

            child_t   = parse_matrix_transform(child.get("transform", ""))
            child_acc = compose_matrices(current_acc, child_t)

            if "BoundaryPolygon" in child.get("class", "").split():
                # Found BoundaryPolygon — collect coords from ALL geometry children
                all_pts: list[tuple[float, float]] = []

                for geom_elem in child:
                    pts = get_element_coords(geom_elem)
                    if pts:
                        if child_acc:
                            pts = apply_matrix(pts, child_acc)
                        all_pts.extend(pts)

                if all_pts:
                    return coords_to_bbox(all_pts)
                # BoundaryPolygon exists but all children had no parseable coords

            # Recurse deeper
            result = search(child, child_acc)
            if result is not None:
                return result

        return None

    return search(elem, base_accumulated)

def extract_furniture_bbox(
    elem: ET.Element,
    accumulated_transform: tuple | None = None,
) -> tuple | None:
    """
    Toilet/Sink: BoundaryPolygon first, recursive all-descendants fallback.

    Two-stage strategy:
    Stage 1 — find_boundary_polygon: recursively searches for a group with
      class="BoundaryPolygon" and returns that group's polygon coordinates.
      This is the cleanest extraction — one precise boundary polygon.

    Stage 2 — collect_recursive fallback: if no BoundaryPolygon is found
      (some furniture elements don't have this wrapper), collect all
      descendant polygon coords with proper per-level transform chains.
      This produces a slightly larger bbox (includes interior detail
      polygons) but is always correct.
    """
    # Stage 1: precise BoundaryPolygon search
    bbox = find_boundary_polygon(elem, accumulated_transform)
    if bbox is not None:
        return bbox

    # Stage 2: robust fallback — all descendants with recursive transforms
    all_coords = collect_recursive(elem, accumulated_transform)
    return coords_to_bbox(all_coords) if all_coords else None

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
    override_width: float | None = None,
    override_height: float | None = None,
) -> tuple[list[str], dict]:
    """
    Parse one SVG file and extract all YOLO-format annotations.

    Uses recursive depth-first traversal to accumulate transforms correctly.
    Each element is processed with the full transform chain from root to
    that element, ensuring furniture positioned by ancestor transforms is
    correctly located.

    Args:
        svg_path: Path to SVG file
        class_id_map: Mapping of class names to IDs
        svg_class_tags: Mapping of class names to SVG keywords
        override_width: If provided, use this width instead of SVG dimensions
        override_height: If provided, use this height instead of SVG dimensions
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

    # ── Use actual image dimensions if provided (fixes alignment) ────────
    if override_width is not None:
        svg_w = override_width
    if override_height is not None:
        svg_h = override_height
    # ───────────────────────────────────────────────────────────────────

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
      1. Get actual image dimensions
      2. Build a unique output filename from category + folder name
      3. Copy PNG → yolo_images_dir
      4. Extract annotations from SVG → write .txt to yolo_labels_dir

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

        # ── GET ACTUAL IMAGE DIMENSIONS (fixes annotation alignment) ──────
        img = cv2.imread(str(img_path))
        if img is None:
            logger.debug(f"Cannot read image: {img_path}")
            agg_stats["skipped_no_img"] += 1
            continue

        actual_h, actual_w = img.shape[:2]
        # ───────────────────────────────────────────────────────────────

        # ── Extract annotations with actual image dimensions ──────────────
        yolo_lines, stats = extract_annotations(
            svg_path, class_id_map, svg_class_tags,
            override_width=float(actual_w),
            override_height=float(actual_h),
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
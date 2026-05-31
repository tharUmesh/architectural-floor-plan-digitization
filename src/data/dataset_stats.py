"""
Dataset Statistics Script — Phase 1, Step 2

PURPOSE:
    Quantify the full CubiCasa5k dataset before subset selection.
    Produces statistics needed for:
    - Informed subset selection decisions
    - Dataset characterization section of the final report
    - Detecting class imbalance early

OUTPUT:
    - data/splits/dataset_stats.json — machine-readable stats
    - Console summary table

HOW TO RUN:
    python src/data/dataset_stats.py

NOTE:
    Run AFTER audit_svg.py. You need to have filled in svg_findings.md
    and updated configs/dataset.yaml with correct svg_class_tags first,
    because this script uses those tag names to count annotations.
"""

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config, load_env, get_env
from src.utils.logger import get_logger

logger = get_logger(__name__)

CATEGORIES = ["colorful", "high_quality", "high_quality_architectural"]


def strip_namespace(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def get_image_size(img_path: Path) -> tuple[int, int] | None:
    """
    Get image dimensions without loading full pixel data.

    Why PIL over OpenCV here: PIL.Image.open() reads only the header
    for common formats, making it much faster when scanning thousands
    of files just for their dimensions.
    """
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def count_annotations_in_svg(
    svg_path: Path,
    svg_class_tags: dict[str, list[str]]
) -> dict[str, int]:
    """
    Count how many instances of each target class exist in one SVG.

    Matching strategy:
      The SVG class attribute holds space-separated words (like CSS).
      e.g. class="Door Swing Beside" → words: ["Door", "Swing", "Beside"]
      We check if our keyword ("Door") is one of those words.
      This correctly matches all door variants without false positives
      like "Doors" (a sub-element inside refrigerators).

    For structural elements (Door, Window, Wall), the id attribute is
    also checked as a fallback, since these carry id="Door" etc.
    """
    counts = {cls: 0 for cls in svg_class_tags}

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        for elem in root.iter():
            tag = strip_namespace(elem.tag)

            # Only process <g> elements — all architectural elements are groups
            if tag != "g":
                continue

            # Get the class attribute and split into individual words
            class_str  = elem.get("class", "")
            class_words = set(class_str.split())   # e.g. {"Door", "Swing", "Beside"}

            # Also check the id attribute as fallback for structural elements
            id_str = elem.get("id", "")

            for class_name, keywords in svg_class_tags.items():
                for keyword in keywords:
                    # Check class words (primary) or id exact match (fallback)
                    if keyword in class_words or id_str == keyword:
                        counts[class_name] += 1
                        break   # Don't double-count if both class and id match

    except ET.ParseError:
        pass

    return counts


def audit_folder(
    folder_path: Path,
    svg_class_tags: dict[str, list[str]],
    image_filename: str
) -> dict | None:
    """
    Examine one CubiCasa5k subfolder and return its statistics.

    Returns None if the folder is invalid (missing files, corrupt SVG,
    image too small). These folders will be excluded from selection.
    """
    svg_path = folder_path / "model.svg"
    img_path = folder_path / image_filename

    # Must have both the image and SVG annotation
    if not svg_path.exists() or not img_path.exists():
        return None

    img_size = get_image_size(img_path)
    if img_size is None:
        return None

    annotation_counts = count_annotations_in_svg(svg_path, svg_class_tags)
    total_annotations = sum(annotation_counts.values())

    return {
        "folder": str(folder_path),
        "category": folder_path.parent.name,
        "image_width": img_size[0],
        "image_height": img_size[1],
        "annotation_counts": annotation_counts,
        "total_annotations": total_annotations,
    }


def compute_summary(folder_stats: list[dict]) -> dict:
    """
    Aggregate per-folder stats into a dataset-level summary.
    This is what goes into the final report.
    """
    total_folders = len(folder_stats)
    per_category = Counter(f["category"] for f in folder_stats)

    # Image size stats
    widths = [f["image_width"] for f in folder_stats]
    heights = [f["image_height"] for f in folder_stats]

    # Annotation counts
    class_totals = defaultdict(int)
    annotation_densities = []  # annotations per image
    zero_annotation_folders = []

    for f in folder_stats:
        for cls, count in f["annotation_counts"].items():
            class_totals[cls] += count
        annotation_densities.append(f["total_annotations"])
        if f["total_annotations"] == 0:
            zero_annotation_folders.append(f["folder"])

    return {
        "total_valid_folders": total_folders,
        "per_category": dict(per_category),
        "image_size": {
            "width":  {"min": min(widths),  "max": max(widths),
                       "mean": round(sum(widths)/len(widths), 1)},
            "height": {"min": min(heights), "max": max(heights),
                       "mean": round(sum(heights)/len(heights), 1)},
        },
        "class_totals": dict(class_totals),
        "annotation_density": {
            "min":  min(annotation_densities),
            "max":  max(annotation_densities),
            "mean": round(sum(annotation_densities)/len(annotation_densities), 2),
        },
        "zero_annotation_count": len(zero_annotation_folders),
        "zero_annotation_folders": zero_annotation_folders[:10],  # sample only
    }


def print_summary_table(summary: dict) -> None:
    """Print a formatted summary table to console."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS SUMMARY")
    print("=" * 60)

    print(f"\nTotal valid folders: {summary['total_valid_folders']}")
    print("\nPer category:")
    for cat, count in summary["per_category"].items():
        pct = count / summary["total_valid_folders"] * 100
        print(f"  {cat:<35} {count:>5}  ({pct:.1f}%)")

    print("\nImage sizes (width):")
    s = summary["image_size"]["width"]
    print(f"  min={s['min']}px  max={s['max']}px  mean={s['mean']}px")

    print("\nImage sizes (height):")
    s = summary["image_size"]["height"]
    print(f"  min={s['min']}px  max={s['max']}px  mean={s['mean']}px")

    print("\nAnnotations per class (across full dataset):")
    total_ann = sum(summary["class_totals"].values())
    for cls, count in sorted(summary["class_totals"].items(),
                              key=lambda x: -x[1]):
        pct = count / total_ann * 100 if total_ann > 0 else 0
        print(f"  {cls:<15} {count:>6}  ({pct:.1f}%)")

    print("\nAnnotation density per image:")
    d = summary["annotation_density"]
    print(f"  min={d['min']}  max={d['max']}  mean={d['mean']}")

    if summary["zero_annotation_count"] > 0:
        print(f"\n⚠ Folders with ZERO annotations: "
              f"{summary['zero_annotation_count']}")
        print("  These will be excluded from subset selection.")

    print("=" * 60)


def main():
    load_env()

    cubicasa_root_str = get_env("CUBICASA_ROOT")
    if not cubicasa_root_str:
        logger.error("CUBICASA_ROOT not set in .env")
        sys.exit(1)

    cubicasa_root = Path(cubicasa_root_str)
    dataset_config = load_config("configs/dataset.yaml")

    svg_class_tags = {
        cls: tags
        for cls, tags in dataset_config["svg_class_tags"].items()
    }
    image_filename = dataset_config["cubicasa"]["image_filename"]
    filters = dataset_config["filters"]

    logger.info("Scanning full dataset — this may take a few minutes...")

    all_folder_stats = []

    for category in CATEGORIES:
        cat_path = cubicasa_root / category
        if not cat_path.exists():
            logger.warning(f"Missing category folder: {cat_path}")
            continue

        subfolders = sorted([
            d for d in cat_path.iterdir() if d.is_dir()
        ])

        logger.info(f"Scanning {len(subfolders)} folders in {category}...")

        for folder in tqdm(subfolders, desc=category, unit="folder"):
            stats = audit_folder(folder, svg_class_tags, image_filename)

            if stats is None:
                continue

            # Apply quality filters from configs/dataset.yaml
            w, h = stats["image_width"], stats["image_height"]
            if w < filters["min_image_width"]:
                continue
            if h < filters["min_image_height"]:
                continue
            if stats["total_annotations"] < filters["min_annotations_per_image"]:
                continue

            all_folder_stats.append(stats)

    summary = compute_summary(all_folder_stats)
    print_summary_table(summary)

    # Save full per-folder stats and summary
    output = {
        "summary": summary,
        "folders": all_folder_stats,
    }

    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    output_path = splits_dir / "dataset_stats.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Full statistics saved to: {output_path}")
    logger.info(f"Total valid folders after filtering: "
                f"{summary['total_valid_folders']}")
    logger.info("Next step: run src/data/select_subset.py")


if __name__ == "__main__":
    main()
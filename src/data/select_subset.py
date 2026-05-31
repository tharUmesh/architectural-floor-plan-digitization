"""
Stratified Subset Selection Script — Phase 1, Step 3

PURPOSE:
    Systematically select a reproducible subset of CubiCasa5k folders
    for training. Uses stratified proportional sampling to ensure all
    three dataset categories are represented in their natural proportions.
    Assigns each folder to train/val/test splits.

OUTPUT:
    - data/splits/selected_folders.json  — the definitive record of
      which folders are used in this experiment
    - data/splits/train.json / val.json / test.json  — per-split lists

HOW TO RUN:
    make select-data
    OR
    python src/data/select_subset.py

REPRODUCIBILITY:
    This script uses seed=42 (from configs/dataset.yaml).
    Running it again on any machine produces the same 500 folders.
    The output JSON is committed to Git as part of the experiment record.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config, load_env, get_env
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_valid_folders(stats_path: Path) -> dict[str, list[str]]:
    """
    Load the pre-computed valid folder list from dataset_stats.json.

    Why load from stats rather than re-scan: dataset_stats.py already
    applied all quality filters. We don't re-apply them here — we trust
    the stats file. This keeps selection fast and consistent.

    Returns dict: {category_name: [list of folder paths]}
    """
    with open(stats_path) as f:
        stats = json.load(f)

    per_category = defaultdict(list)
    for folder_info in stats["folders"]:
        per_category[folder_info["category"]].append(folder_info["folder"])

    logger.info("Valid folders loaded from stats:")
    for cat, folders in per_category.items():
        logger.info(f"  {cat}: {len(folders)} folders")

    return dict(per_category)


def proportional_counts(
    per_category: dict[str, list[str]],
    total: int
) -> dict[str, int]:
    """
    Calculate how many folders to select from each category,
    proportional to the category's share of the total valid pool.

    Example:
        colorful=276, high_quality=992, high_quality_architectural=3732
        total valid = 5000, target = 500
        colorful gets: round(276/5000 * 500) = 28
        high_quality gets: round(992/5000 * 500) = 99
        high_quality_architectural gets: round(3732/5000 * 500) = 373

    We round each count and then adjust the last category to ensure
    the total is exactly `total` (rounding can cause off-by-one errors).
    """
    total_available = sum(len(v) for v in per_category.values())
    counts = {}

    for cat, folders in per_category.items():
        proportion = len(folders) / total_available
        counts[cat] = round(proportion * total)

    # Fix rounding error: adjust the largest category up or down
    diff = total - sum(counts.values())
    largest_cat = max(counts, key=counts.__getitem__)
    counts[largest_cat] += diff

    logger.info(f"Proportional selection plan (target={total}):")
    for cat, count in counts.items():
        pct = count / total * 100
        logger.info(f"  {cat}: {count} folders ({pct:.1f}%)")

    return counts


def stratified_select(
    per_category: dict[str, list[str]],
    counts: dict[str, int],
    seed: int
) -> dict[str, list[str]]:
    """
    Perform the actual random sampling within each category.

    Why sort before sampling: Ensures deterministic ordering before
    random.sample applies its seed. Without sorting, different OS file
    system orderings could produce different results even with the same seed.
    """
    random.seed(seed)
    selected = {}

    for cat, folders in per_category.items():
        n = counts.get(cat, 0)
        if n > len(folders):
            logger.warning(
                f"{cat}: requested {n} but only {len(folders)} available. "
                f"Using all {len(folders)}."
            )
            n = len(folders)
        selected[cat] = random.sample(sorted(folders), n)

    return selected


def assign_splits(
    selected: dict[str, list[str]],
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> dict[str, list[str]]:
    """
    Assign each selected folder to train, val, or test split.

    Strategy: Stratified assignment — within each category, folders are
    shuffled (with seed) then split by ratio. This guarantees each split
    contains a representative mix of all three categories.

    Returns dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    random.seed(seed + 1)  # Different seed from selection to avoid correlation

    splits = {"train": [], "val": [], "test": []}

    for cat, folders in selected.items():
        shuffled = folders.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = round(n * train_ratio)
        n_val   = round(n * val_ratio)
        # Test gets the remainder
        n_test  = n - n_train - n_val

        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train:n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val:])

        logger.info(
            f"{cat} split → "
            f"train:{n_train}  val:{n_val}  test:{n_test}"
        )

    return splits


def save_splits(
    selected: dict[str, list[str]],
    splits: dict[str, list[str]],
    config: dict,
    splits_dir: Path
) -> None:
    """
    Save the selection and split assignments to JSON files.

    Files saved:
    - selected_folders.json  : complete record with metadata
    - train.json / val.json / test.json : per-split folder lists

    The metadata (config snapshot, counts) makes the file self-describing —
    you can read it six months later and understand exactly what it contains.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ── Complete selection record (the authoritative file) ────────────────
    selection_record = {
        "metadata": {
            "total_selected": sum(len(v) for v in selected.values()),
            "seed": config["selection"]["seed"],
            "strategy": config["selection"]["strategy"],
            "per_category": {
                cat: len(folders) for cat, folders in selected.items()
            },
            "splits": {
                split: len(folders) for split, folders in splits.items()
            },
        },
        "selected_by_category": selected,
        "splits": splits,
    }

    master_path = splits_dir / "selected_folders.json"
    with open(master_path, "w") as f:
        json.dump(selection_record, f, indent=2)
    logger.info(f"Master selection saved → {master_path}")

    # ── Individual split files ────────────────────────────────────────────
    for split_name, folder_list in splits.items():
        split_path = splits_dir / f"{split_name}.json"
        with open(split_path, "w") as f:
            json.dump(
                {"split": split_name, "count": len(folder_list),
                 "folders": folder_list},
                f, indent=2
            )
        logger.info(
            f"{split_name}.json saved → {len(folder_list)} folders"
        )


def print_final_summary(splits: dict[str, list[str]]) -> None:
    total = sum(len(v) for v in splits.values())
    print("\n" + "=" * 50)
    print("SUBSET SELECTION COMPLETE")
    print("=" * 50)
    for split, folders in splits.items():
        pct = len(folders) / total * 100
        print(f"  {split:<8}: {len(folders):>4} folders  ({pct:.1f}%)")
    print(f"  {'TOTAL':<8}: {total:>4} folders")
    print("=" * 50)
    print("\nOutput files:")
    print("  data/splits/selected_folders.json  ← commit this to Git")
    print("  data/splits/train.json")
    print("  data/splits/val.json")
    print("  data/splits/test.json")
    print("\nNext step: run src/data/audit_svg.py (if not done)")
    print("           then src/data/convert_annotations.py")


def main():
    load_env()

    stats_path = Path("data/splits/dataset_stats.json")
    if not stats_path.exists():
        logger.error(
            "dataset_stats.json not found. "
            "Run src/data/dataset_stats.py first."
        )
        sys.exit(1)

    config = load_config("configs/dataset.yaml")
    selection_cfg = config["selection"]
    split_cfg = config["split"]

    per_category = load_valid_folders(stats_path)

    counts = proportional_counts(per_category, total=selection_cfg["total"])

    selected = stratified_select(
        per_category, counts, seed=selection_cfg["seed"]
    )

    splits = assign_splits(
        selected,
        train_ratio=split_cfg["train"],
        val_ratio=split_cfg["val"],
        seed=selection_cfg["seed"],
    )

    save_splits(selected, splits, config, splits_dir=Path("data/splits"))
    print_final_summary(splits)


if __name__ == "__main__":
    main()
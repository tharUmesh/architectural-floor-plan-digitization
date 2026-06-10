"""
Systematic Hyperparameter Tuning — Phase 5 Extension

GOAL: mAP50 ≥ 0.85 AND mAP50-95 ≥ 0.60 with fastest inference.

STRATEGY (OFAT — One Factor At A Time):
  1. Test each lever independently to measure its isolated contribution
  2. Combine only the levers that improve results
  3. Final experiment combines all proven improvements

CURRENT BASELINE (YOLO11L):
  mAP50=0.838  mAP50-95=0.571
  Weak classes: Toilet=0.705  Staircase=0.760

EXPERIMENTS:
  E01 — Resolution 1280  (biggest expected jump for small objects)
  E02 — copy_paste=0.60  (rare class oversampling)
  E03 — cls=2.5          (sharper classification)
  E04 — box=10.0         (tighter boxes → mAP50-95)
  E05 — degrees=90       (orientation invariance)
  E06 — E01+E02+E03      (combined minority class fix)
  E07 — E06+E04+E05      (full combination)
  E08 — YOLO11X + E07    (architecture ceiling — run if E07 misses target)

OUTPUTS:
  models/tuning_runs/<exp_id>/weights/best.pt
  models/tuning_runs/tuning_results.csv
  models/tuning_runs/tuning_summary.txt

HOW TO RUN:
  python src/training/hyperparameter_tuning.py
  python src/training/hyperparameter_tuning.py --only E01 E06 E07
  python src/training/hyperparameter_tuning.py --resume   (skip completed)
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these paths if your layout differs
# ─────────────────────────────────────────────────────────────────────────────

# Try both common dataset config paths automatically
_CANDIDATE_DATA_YAMLS = [
    "data/floorplan.yaml",
    "data/yolo_dataset_processed/dataset.yaml",
    "data/yolo_dataset/dataset.yaml",
]

RUNS_DIR     = Path("models/tuning_runs")
RESULTS_CSV  = RUNS_DIR / "tuning_results.csv"
SUMMARY_FILE = RUNS_DIR / "tuning_summary.txt"

# Targets from project requirements
TARGET_MAP50    = 0.85
TARGET_MAP5095  = 0.60

# Class names (must match your dataset.yaml order)
CLASS_NAMES = ["door", "window", "wall", "staircase", "toilet", "sink"]


# ─────────────────────────────────────────────────────────────────────────────
# Default parameters shared across all experiments
# (these replicate the current YOLO11L baseline behaviour)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "model":         "yolo11l.pt",
    "imgsz":         640,
    "batch":         32,
    "epochs":        150,
    "patience":      40,
    "workers":       8,
    "device":        0,
    "amp":           True,
    "lr0":           0.0008,
    "lrf":           0.01,
    "cos_lr":        True,
    "warmup_epochs": 5,
    "cls":           1.5,
    "box":           7.5,
    "mosaic":        1.0,
    "close_mosaic":  15,
    "copy_paste":    0.30,
    "mixup":         0.0,
    "degrees":       0.0,
    "fliplr":        0.5,
    "flipud":        0.5,
    "scale":         0.30,
    "translate":     0.20,
    "hsv_h":         0.0,
    "hsv_s":         0.0,
    "hsv_v":         0.1,
    "erasing":       0.0,
    "save_period":   10,
    "val":           True,
}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment definitions
# Each entry overrides only the parameters that differ from DEFAULTS.
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    # ── Group 1: Resolution ────────────────────────────────────────────────
    {
        "id":          "E01_res_1280",
        "group":       "G1_Resolution",
        "hypothesis":  "imgsz 640→1280: small objects (Toilet, Staircase) need more pixels",
        "overrides": {
            "imgsz": 1280,
            "batch": 8,   # 32GB VRAM comfortable at 1280px with batch=8
        },
    },

    # ── Group 2: Minority class handling ──────────────────────────────────
    {
        "id":          "E02_copy_paste_high",
        "group":       "G2_Minority_Class",
        "hypothesis":  "copy_paste 0.30→0.60: more synthetic Toilet/Staircase samples",
        "overrides": {
            "copy_paste": 0.60,
        },
    },
    {
        "id":          "E03_cls_weight_high",
        "group":       "G2_Minority_Class",
        "hypothesis":  "cls 1.5→2.5: larger loss penalty forces sharper class discrimination",
        "overrides": {
            "cls": 2.5,
        },
    },

    # ── Group 3: Localization (mAP50-95) ──────────────────────────────────
    {
        "id":          "E04_box_weight",
        "group":       "G3_Localization",
        "hypothesis":  "box 7.5→10.0: tighter bboxes → better mAP50-95",
        "overrides": {
            "box": 10.0,
        },
    },

    # ── Group 4: Augmentation ─────────────────────────────────────────────
    {
        "id":          "E05_rotation_90",
        "group":       "G4_Augmentation",
        "hypothesis":  "degrees=90: floor plans exist in any orientation",
        "overrides": {
            "degrees": 90.0,
        },
    },

    # ── Group 5: Combined experiments ─────────────────────────────────────
    {
        "id":          "E06_minority_combo",
        "group":       "G5_Combined",
        "hypothesis":  "E01+E02+E03: all minority-class fixes stacked",
        "overrides": {
            "imgsz":      1280,
            "batch":      8,
            "copy_paste": 0.50,   # slightly conservative vs 0.60 for stability
            "cls":        2.5,
        },
    },
    {
        "id":          "E07_full_combo",
        "group":       "G5_Combined",
        "hypothesis":  "E06+E04+E05: full combination of all proven improvements",
        "overrides": {
            "imgsz":      1280,
            "batch":      8,
            "copy_paste": 0.50,
            "cls":        2.5,
            "box":        10.0,
            "degrees":    90.0,
        },
    },

    # ── Group 6: Architecture (only run if E07 misses targets) ────────────
    {
        "id":          "E08_yolo11x_combo",
        "group":       "G6_Architecture",
        "hypothesis":  "YOLO11X + best settings: architecture ceiling check",
        "conditional": True,   # only run if E07 doesn't hit both targets
        "overrides": {
            "model":      "yolo11x.pt",
            "imgsz":      1280,
            "batch":      4,    # YOLO11X is larger; reduce batch for VRAM safety
            "copy_paste": 0.50,
            "cls":        2.5,
            "box":        10.0,
            "degrees":    90.0,
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Results tracking
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "id", "group", "model", "imgsz", "cls", "box",
    "copy_paste", "degrees",
    "precision", "recall",
    "mAP50", "mAP50_95",
    "AP_door", "AP_window", "AP_wall",
    "AP_staircase", "AP_toilet", "AP_sink",
    "inference_ms_per_image",
    "training_time_hrs",
    "hits_map50_target", "hits_map5095_target", "hits_both_targets",
    "timestamp",
]


def load_existing_results() -> dict[str, dict]:
    """Load previously saved results (for resume support)."""
    existing = {}
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                existing[row["id"]] = row
    return existing


def append_result(row: dict) -> None:
    """Append one experiment result to the CSV file."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Inference time measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_inference_time(model: YOLO, data_yaml: str, n_warmup: int = 10) -> float:
    """
    Measure mean inference time in ms/image on the validation set.

    Runs n_warmup predictions that are discarded (GPU warmup),
    then times the rest and returns mean ms/image.
    """
    from ultralytics.data import build_dataloader
    import numpy as np

    # Gather test image paths from the YOLO dataset
    val_results = model.val(data=data_yaml, split="val", verbose=False)

    # Use YOLO's built-in speed reporting (it measures preprocessing + inference)
    # speed dict has keys: preprocess, inference, postprocess (all in ms/image)
    speed = val_results.speed
    total_ms = speed.get("preprocess", 0) + speed.get("inference", 0) + speed.get("postprocess", 0)
    return round(total_ms, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    exp:           dict,
    data_yaml:     str,
    existing:      dict[str, dict],
    resume:        bool,
) -> dict | None:
    """
    Run one experiment end-to-end:
      1. Merge override params with defaults
      2. Train the model
      3. Evaluate on test set
      4. Measure inference speed
      5. Save result row

    Returns the result dict, or None if skipped.
    """
    exp_id = exp["id"]

    # ── Resume check ─────────────────────────────────────────────────────
    if resume and exp_id in existing:
        logger.info(f"[{exp_id}] Already completed — skipping (--resume mode)")
        return existing[exp_id]

    # ── Build full param dict ─────────────────────────────────────────────
    params   = {**DEFAULTS, **exp.get("overrides", {})}
    model_pt = params.pop("model")  # handled separately

    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {exp_id}")
    logger.info(f"Group:      {exp['group']}")
    logger.info(f"Hypothesis: {exp['hypothesis']}")
    logger.info(f"Model:      {model_pt}")
    logger.info(f"Overrides:  {exp.get('overrides', {})}")
    logger.info(f"{'='*60}")

    # ── Train ─────────────────────────────────────────────────────────────
    model = YOLO(model_pt)
    t_start = time.time()

    model.train(
        data    = data_yaml,
        project = str(RUNS_DIR),
        name    = exp_id,
        **params,
    )

    train_hrs = (time.time() - t_start) / 3600
    logger.info(f"[{exp_id}] Training finished in {train_hrs:.2f}h")

    # ── Evaluate on TEST split ────────────────────────────────────────────
    best_weights = RUNS_DIR / exp_id / "weights" / "best.pt"
    if not best_weights.exists():
        logger.error(f"[{exp_id}] No best.pt found — evaluation skipped")
        return None

    eval_model = YOLO(str(best_weights))
    metrics    = eval_model.val(data=data_yaml, split="test", verbose=False)

    # ── Inference time ────────────────────────────────────────────────────
    inf_ms = measure_inference_time(eval_model, data_yaml)

    # ── Build result row ──────────────────────────────────────────────────
    ap_per_class = metrics.box.ap_class_index  # indices into ap array
    ap_values    = metrics.box.ap              # AP values per class

    # Map class index → AP (handle cases where a class may be absent)
    ap_by_name = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        if i < len(ap_values):
            ap_by_name[cls_name] = round(float(ap_values[i]), 4)
        else:
            ap_by_name[cls_name] = 0.0

    map50    = round(float(metrics.box.map50), 4)
    map5095  = round(float(metrics.box.map),   4)

    row = {
        "id":                     exp_id,
        "group":                  exp["group"],
        "model":                  model_pt,
        "imgsz":                  exp.get("overrides", {}).get("imgsz", DEFAULTS["imgsz"]),
        "cls":                    exp.get("overrides", {}).get("cls",   DEFAULTS["cls"]),
        "box":                    exp.get("overrides", {}).get("box",   DEFAULTS["box"]),
        "copy_paste":             exp.get("overrides", {}).get("copy_paste", DEFAULTS["copy_paste"]),
        "degrees":                exp.get("overrides", {}).get("degrees", DEFAULTS["degrees"]),
        "precision":              round(float(metrics.box.mp), 4),
        "recall":                 round(float(metrics.box.mr), 4),
        "mAP50":                  map50,
        "mAP50_95":               map5095,
        "AP_door":                ap_by_name.get("door",       0.0),
        "AP_window":              ap_by_name.get("window",     0.0),
        "AP_wall":                ap_by_name.get("wall",       0.0),
        "AP_staircase":           ap_by_name.get("staircase",  0.0),
        "AP_toilet":              ap_by_name.get("toilet",     0.0),
        "AP_sink":                ap_by_name.get("sink",       0.0),
        "inference_ms_per_image": inf_ms,
        "training_time_hrs":      round(train_hrs, 2),
        "hits_map50_target":      map50   >= TARGET_MAP50,
        "hits_map5095_target":    map5095 >= TARGET_MAP5095,
        "hits_both_targets":      map50 >= TARGET_MAP50 and map5095 >= TARGET_MAP5095,
        "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    append_result(row)
    _print_experiment_result(row)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_experiment_result(row: dict) -> None:
    """Print a compact result card after each experiment."""
    hit = "✓" if row["hits_both_targets"] else "✗"
    print(f"\n  ┌── {row['id']} [{hit}]")
    print(f"  │  mAP50={row['mAP50']:.4f} (target≥{TARGET_MAP50})  "
          f"mAP50-95={row['mAP50_95']:.4f} (target≥{TARGET_MAP5095})")
    print(f"  │  Toilet={row['AP_toilet']:.4f}  Staircase={row['AP_staircase']:.4f}")
    print(f"  └── inference={row['inference_ms_per_image']}ms  "
          f"train={row['training_time_hrs']:.1f}h")


def print_final_summary(all_results: list[dict]) -> None:
    """Print and save the full comparison table at the end."""

    # Sort by mAP50 descending
    ranked = sorted(all_results, key=lambda r: float(r["mAP50"]), reverse=True)

    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("HYPERPARAMETER TUNING — FINAL SUMMARY")
    lines.append(f"Baseline YOLO11L:  mAP50=0.8381  mAP50-95=0.5707")
    lines.append(f"Target:            mAP50≥{TARGET_MAP50}  mAP50-95≥{TARGET_MAP5095}")
    lines.append("=" * 100)

    header = (
        f"{'Rank':<5} {'ID':<28} {'mAP50':>6} {'mAP50-95':>9} "
        f"{'Toilet':>7} {'Stairs':>7} "
        f"{'Inf(ms)':>8} {'Hrs':>5} {'Hit?':>5}"
    )
    lines.append(header)
    lines.append("-" * 100)

    for rank, r in enumerate(ranked, 1):
        hit = "BOTH✓" if r["hits_both_targets"] else (
              "50✓  " if r["hits_map50_target"] else
              " 95✓ " if r["hits_map5095_target"] else "  ✗  ")

        line = (
            f"{rank:<5} {r['id']:<28} "
            f"{float(r['mAP50']):>6.4f} {float(r['mAP50_95']):>9.4f} "
            f"{float(r['AP_toilet']):>7.4f} {float(r['AP_staircase']):>7.4f} "
            f"{float(r['inference_ms_per_image']):>8.1f} "
            f"{float(r['training_time_hrs']):>5.1f} "
            f"{hit:>5}"
        )
        lines.append(line)

    lines.append("=" * 100)

    # Best model recommendation
    best = ranked[0]
    lines.append(f"\nBEST RESULT: {best['id']}")
    lines.append(f"  mAP50={best['mAP50']}  mAP50-95={best['mAP50_95']}")
    lines.append(f"  Improvement over baseline:")
    lines.append(f"    mAP50:    {float(best['mAP50']) - 0.8381:+.4f}")
    lines.append(f"    mAP50-95: {float(best['mAP50_95']) - 0.5707:+.4f}")
    lines.append(f"  Weakest class remaining: "
                 f"Toilet={best['AP_toilet']}, Staircase={best['AP_staircase']}")
    lines.append(f"\n  Best weights: models/tuning_runs/{best['id']}/weights/best.pt")

    # Per-class analysis across all experiments
    lines.append("\nPER-CLASS IMPROVEMENT TRACKING (vs YOLO11L baseline):")
    baseline = {"door": 0.9275, "window": 0.9062, "wall": 0.8378,
                "staircase": 0.7603, "toilet": 0.7048, "sink": 0.8922}
    lines.append(f"  {'Class':<12} {'Baseline':>8} {'Best':>8} {'Delta':>8} {'Best Exp':<30}")
    for cls in CLASS_NAMES:
        col = f"AP_{cls}"
        vals = [(float(r[col]), r["id"]) for r in all_results if col in r]
        if not vals:
            continue
        best_val, best_exp = max(vals, key=lambda x: x[0])
        delta = best_val - baseline.get(cls, 0)
        lines.append(
            f"  {cls:<12} {baseline.get(cls, 0):>8.4f} "
            f"{best_val:>8.4f} {delta:>+8.4f}   {best_exp}"
        )

    summary = "\n".join(lines)
    print(summary)

    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    logger.info(f"Summary saved → {SUMMARY_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def find_data_yaml() -> str:
    """Auto-detect the dataset YAML path from known candidates."""
    for candidate in _CANDIDATE_DATA_YAMLS:
        if Path(candidate).exists():
            logger.info(f"Dataset YAML: {candidate}")
            return candidate
    logger.error(
        "No dataset YAML found. Checked:\n" +
        "\n".join(f"  {p}" for p in _CANDIDATE_DATA_YAMLS)
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Systematic hyperparameter tuning")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="EXP_ID",
        help="Run only specific experiments by ID, e.g. --only E01 E06 E07",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments that already have results in tuning_results.csv",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print experiment list and exit without training",
    )
    args = parser.parse_args()

    # ── Print experiment list if requested ───────────────────────────────
    if args.list:
        print(f"\n{'ID':<28} {'Group':<25} Hypothesis")
        print("-" * 90)
        for exp in EXPERIMENTS:
            cond = " [conditional]" if exp.get("conditional") else ""
            print(f"{exp['id']:<28} {exp['group']:<25} {exp['hypothesis'][:45]}{cond}")
        return

    # ── GPU check ────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Run on the Ubuntu RTX 5090 machine.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_gb   = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {gpu_name} ({gpu_gb:.1f} GB)")

    # ── Find dataset YAML ────────────────────────────────────────────────
    data_yaml = find_data_yaml()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load existing results for resume ─────────────────────────────────
    existing = load_existing_results()
    if existing:
        logger.info(f"Loaded {len(existing)} existing results from {RESULTS_CSV}")

    # ── Filter experiment list ────────────────────────────────────────────
    exps_to_run = EXPERIMENTS
    if args.only:
        exps_to_run = [e for e in EXPERIMENTS if e["id"] in args.only]
        logger.info(f"Running {len(exps_to_run)} selected experiments: {args.only}")

    # ── Run experiments ───────────────────────────────────────────────────
    all_results = list(existing.values())
    targets_hit = False

    for exp in exps_to_run:
        # Skip conditional experiments if targets already met
        if exp.get("conditional") and targets_hit:
            logger.info(
                f"[{exp['id']}] Skipping — conditional experiment "
                f"(targets already achieved)"
            )
            continue

        result = run_experiment(exp, data_yaml, existing, args.resume)

        if result is not None:
            all_results.append(result)
            if result["hits_both_targets"]:
                targets_hit = True
                logger.info(
                    f"[{exp['id']}] 🎯 BOTH TARGETS HIT! "
                    f"mAP50={result['mAP50']}  mAP50-95={result['mAP50_95']}"
                )

    # ── Final summary ─────────────────────────────────────────────────────
    if all_results:
        print_final_summary(all_results)
    else:
        logger.warning("No results to summarise.")


if __name__ == "__main__":
    main()
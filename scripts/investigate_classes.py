"""
One-off diagnostic script to find the actual SVG class names used
for staircase-related elements across the dataset.

Run with: python scripts/investigate_classes.py
"""

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_env, get_env

load_env()

CUBICASA_ROOT = Path(get_env("CUBICASA_ROOT"))
STATS_PATH    = Path("data/splits/dataset_stats.json")

# ── Load the folder list we already computed ─────────────────────────────────
with open(STATS_PATH) as f:
    stats = json.load(f)

all_folders = [Path(entry["folder"]) for entry in stats["folders"]]

# ── Search for any class containing these substrings (case-insensitive) ──────
SEARCH_TERMS = ["stair", "step", "ladder", "ramp"]

found_classes   = Counter()   # full class strings that matched
found_ids       = Counter()   # id attribute values that matched
sample_elements = []          # a few raw XML snippets for manual inspection

print(f"Searching {len(all_folders)} SVG files...")
print("Looking for class/id attributes containing: " + str(SEARCH_TERMS))
print()

for folder in all_folders:
    svg_path = folder / "model.svg"
    if not svg_path.exists():
        continue

    try:
        tree = ET.parse(svg_path)
    except ET.ParseError:
        continue

    for elem in tree.getroot().iter():
        tag        = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        class_str  = elem.get("class", "").lower()
        id_str     = elem.get("id", "").lower()

        for term in SEARCH_TERMS:
            if term in class_str:
                found_classes[elem.get("class", "")] += 1
                # Save a few examples to look at
                if len(sample_elements) < 10:
                    sample_elements.append({
                        "tag": tag,
                        "class": elem.get("class", ""),
                        "id": elem.get("id", ""),
                        "folder": str(folder.name)
                    })
            if term in id_str:
                found_ids[elem.get("id", "")] += 1

# ── Report ────────────────────────────────────────────────────────────────────
if not found_classes and not found_ids:
    print("NO MATCHES FOUND for any staircase-related term.")
    print("The dataset may genuinely have no staircase annotations.")
    print("Recommendation: Remove 'Staircase' from target classes.")
else:
    print("MATCHING CLASS ATTRIBUTE VALUES (full strings):")
    for class_val, count in found_classes.most_common(20):
        print(f"  {count:>5}×  class='{class_val}'")

    if found_ids:
        print("\nMATCHING ID ATTRIBUTE VALUES:")
        for id_val, count in found_ids.most_common(10):
            print(f"  {count:>5}×  id='{id_val}'")

    print("\nSAMPLE XML ELEMENTS:")
    for s in sample_elements:
        print(f"  <{s['tag']} class=\"{s['class']}\" id=\"{s['id']}\">  "
              f"(folder: {s['folder']})")
"""Build manifest.csv (image,label,sublabel) from the ARMBench Image defect folder.

Each image data/<id>/<id>_<n>.jpg has a sibling .json like
    {"id": "..._1.jpg", "label": "nominal", "sublabel": [], "polygon": []}

Usage:
    python3 00_make_manifest.py /path/to/armbench-defects-image-0.1 --per-class 3000

Output labels: multi_pick, package_defect, clean.
The script ALWAYS prints every raw label and sublabel it saw. If the mapping
looks wrong, edit map_label() below and re-run.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict


def map_label(raw):
    s = str(raw).lower().replace("-", "_").replace(" ", "_")
    if "multi" in s:
        return "multi_pick"
    if s in ("nominal", "clean", "normal", "success"):
        return "clean"
    return "package_defect"  # every other defect type (open, deformed, damaged, ...)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--out", default="manifest.csv")
    args = ap.parse_args()

    data = os.path.join(args.root, "data")
    raw_labels, raw_subs = Counter(), Counter()
    by_label = defaultdict(list)
    missing = bad = 0
    for dirpath, _, files in os.walk(data):
        for f in files:
            if not f.endswith(".json"):
                continue
            jp = os.path.join(dirpath, f)
            img = jp[:-5] + ".jpg"
            if not os.path.exists(img):
                missing += 1
                continue
            try:
                j = json.load(open(jp))
            except Exception:
                bad += 1
                continue
            raw = j.get("label", "")
            subs = j.get("sublabel") or []
            if isinstance(subs, str):
                subs = [subs]
            raw_labels[raw] += 1
            for s in subs:
                raw_subs[str(s)] += 1
            by_label[map_label(raw)].append((img, "|".join(str(s) for s in subs)))

    print("\nRaw 'label' values in the JSON files:")
    for k, v in raw_labels.most_common():
        print(f"  {str(k):30s} {v:7d}  -> {map_label(k)}")
    print("\nRaw 'sublabel' values:")
    for k, v in raw_subs.most_common(30):
        print(f"  {k:30s} {v:7d}")
    print(f"\nJSON without a .jpg: {missing}   unreadable JSON: {bad}")

    print("\nMapped label counts:")
    for lab, items in by_label.items():
        print(f"  {lab:15s} {len(items)}   e.g. {items[0][0]}")

    random.seed(0)
    rows = []
    for lab, items in by_label.items():
        random.shuffle(items)
        rows += [(p, lab, sub) for p, sub in items[: args.per_class]]
    random.shuffle(rows)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "label", "sublabel"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

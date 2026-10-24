"""Failure modes taken from ARMBench's own labels (label + sublabel in each image's .json)."""
import json
import os

NAMES = {
    "multi_pick": "Multi-pick (two or more items lifted)",
    "open_book_jacket": "Open book jacket",
    "open_box": "Open box",
    "book_jacket": "Book jacket",
    "open_book": "Open book",
    "partial_box": "Partial box",
    "bag_damage": "Torn or empty bag",
    "other_damage": "Crushed box and other damage",
}

# Only multi-pick is a physical event the simulator can recreate. The three scene
# recipes below are the causes P3/P2 build; they are our hypotheses, not data labels.
SIMULATED = {
    "multi_pick": [
        {"id": "flat_stacked", "name": "Flat items stacked"},
        {"id": "small_beside_big", "name": "Small item beside a large one"},
        {"id": "against_wall", "name": "Item against the tote wall"},
    ]
}


def _sub(v):
    if isinstance(v, list):
        return str(v[0]) if v else ""
    return str(v or "")


def mode_of(label, sublabel):
    label = str(label).lower().replace("-", "_")
    if "multi" in label:
        return "multi_pick"
    s = _sub(sublabel).lower().replace("-", "_")
    if s in ("open_book_jacket", "open_box", "book_jacket", "open_book", "partial_box"):
        return s
    if s in ("torn_bag", "empty_bag"):
        return "bag_damage"
    return "other_damage"


def mode_for_image(path, fallback_label="package_defect"):
    jp = os.path.splitext(path)[0] + ".json"
    try:
        j = json.load(open(jp))
        return mode_of(j.get("label", fallback_label), j.get("sublabel"))
    except Exception:
        return mode_of(fallback_label, "")

"""Robustness check: make the simulator's failure rules stricter than the ones ClearPick's risk scoring assumes.

ClearPick's scorer (brain/score.py) mirrors the simulator's thresholds, so a fair question is whether it only
wins because it knows them exactly. This runs every scene with both robots (no video) after the policies are
loaded, then tightens the simulator's thresholds only; the brain keeps the values it was imported with.

    python -m sim.stress_rules                 # all three settings
    python -m sim.stress_rules --setting 2x    # one setting
"""
import argparse
import json
import tempfile
from pathlib import Path

import sim.run_batch as rb

SETTINGS = {  # (cling overlap, neighbour drag gap m, wall seal margin m)
    "as_assumed": (0.5, 0.010, 0.015),
    "1.5x": (0.4, 0.015, 0.022),
    "2x": (0.3, 0.020, 0.030),
}


def run(setting, scenes_root="scenes", policies=("grab_only", "clearpick")):
    pols = {p: rb.load_policy(p) for p in policies}   # load first: the brain keeps the original thresholds
    import sim.sim as S
    saved = (S.CLING_MIN_OVERLAP, S.NEIGHBOUR_DRAG_GAP, S.WALL_SEAL_MARGIN)
    S.CLING_MIN_OVERLAP, S.NEIGHBOUR_DRAG_GAP, S.WALL_SEAL_MARGIN = SETTINGS[setting]
    try:
        totals, out = {}, tempfile.mkdtemp()
        for f in sorted(Path(scenes_root).glob("*/*.json")):
            if f.parent.name == "mock":
                continue
            scene = json.loads(f.read_text())
            for name, decide in pols.items():
                s = rb.run_scene(scene, name, decide, out, video=False)["summary"]
                t = totals.setdefault((scene.get("mode"), name), [0, 0])
                t[0] += s["double_grabs"] + s["misses"]
                t[1] += s["clean"] + s["double_grabs"] + s["misses"]
        return totals
    finally:
        S.CLING_MIN_OVERLAP, S.NEIGHBOUR_DRAG_GAP, S.WALL_SEAL_MARGIN = saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=list(SETTINGS), help="default: all")
    a = ap.parse_args()
    for setting in [a.setting] if a.setting else list(SETTINGS):
        totals = run(setting)
        failed = {p: sum(v[0] for (m, q), v in totals.items() if q == p) for p in ("grab_only", "clearpick")}
        print(f"{setting:10s} failed picks: grab_only {failed['grab_only']:4d}   clearpick {failed['clearpick']:4d}")
        for (mode, pol), (f, n) in sorted(totals.items()):
            print(f"    {mode:18s} {pol:10s} {f:3d} of {n}")

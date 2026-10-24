"""Check generated scenes before a batch run: required fields, objects inside the tote, no overlaps.

    python -m sim.validate_scenes                 # every scenes/*/*.json
    python -m sim.validate_scenes scenes/flat_stacked

Overlapping objects get shoved apart when physics starts, so any object that moves more than
MAX_SETTLE_MOVE while settling is reported. Exit code 1 if any scene has a problem (Kiro hooks show the
message to the agent).
"""
import json
import sys
from pathlib import Path

import numpy as np

from sim.sim import TOTE_DEFAULT, Sim

REQUIRED = ("id", "size", "pos")
MAX_SETTLE_MOVE = 0.02


def check_scene(scene) -> list:
    problems = []
    for o in scene.get("objects", []):
        missing = [k for k in REQUIRED if k not in o]
        if missing:
            problems.append(f"{o.get('id', '?')}: missing {', '.join(missing)}")
    if problems:
        return problems
    L, W, _ = {**TOTE_DEFAULT, **scene.get("tote", {})}["size"]
    for o in scene["objects"]:
        x, y, _ = o["pos"]
        if abs(x) > L / 2 or abs(y) > W / 2:
            problems.append(f"{o['id']}: centre {o['pos'][:2]} is outside the tote")
    if problems:
        return problems
    try:
        state = Sim(width=64, height=48, video=False).load(scene)
    except Exception as e:  # compile errors, bad shapes
        return [f"does not load: {e}"]
    for o in scene["objects"]:
        now = np.array(state["objects"][o["id"]]["pos"])
        moved = float(np.linalg.norm(now - np.array(o["pos"])))
        if moved > MAX_SETTLE_MOVE:
            problems.append(f"{o['id']}: moved {moved * 100:.1f} cm while settling (overlap or floating?)")
        if abs(now[0]) > L / 2 or abs(now[1]) > W / 2 or now[2] < -0.005:
            problems.append(f"{o['id']}: ended outside the tote at {now.round(3).tolist()}")
    return problems


def main(root="scenes"):
    files = sorted(Path(root).rglob("*.json"))
    bad = 0
    for f in files:
        problems = check_scene(json.loads(f.read_text()))
        if problems:
            bad += 1
            print(f"{f}:", *problems, sep="\n  ", file=sys.stderr)
    print(f"{len(files) - bad}/{len(files)} scenes OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))

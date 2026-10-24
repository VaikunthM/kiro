"""Explain what a policy does on one scene in the REAL simulator.

    python -m brain.diagnose scenes/flat_stacked/1.json clearpick
    python -m brain.diagnose scenes/flat_stacked/1.json grab_only

Prints each object's bbox, scorer features and risks, then every step: the action, the risk, the features of the
chosen item, what disappeared after a grasp (more than one item = a double), and how far a push moved the item.
"""
import importlib
import json
import sys

from .geometry import normalize
from .score import GRASPS, features, score_grasp


def _ids(state):
    return {o["id"]: o for o in normalize(state)["objects"]}


def _fmt(f):
    return " ".join(f"{k}={v:.2f}" for k, v in f.items() if v > 0.005) or "(all zero)"


def trace(scene, decide, sim):
    state = sim.load(scene)
    state = sim.state()
    n = normalize(state)
    print(f"scene {n['scene_id']} mode={n['mode']!r} objects={len(n['objects'])}")
    for o in n["objects"]:
        risks = " ".join(f"{g}={score_grasp(n, o['id'], g):.2f}" for g in GRASPS)
        print(f"  {o['id']:18} z {o['lo'][2]:.3f}-{o['hi'][2]:.3f}  xy [{o['lo'][0]:+.3f},{o['hi'][0]:+.3f}]x[{o['lo'][1]:+.3f},{o['hi'][1]:+.3f}]  {risks}")
        print(f"  {'':18} {_fmt(features(n, o['id']))}")
    print()
    for i in range(3 * len(n["objects"]) + 6):
        state = sim.state()
        step = decide(state)
        if step is None or not normalize(state)["objects"]:
            break
        before = _ids(state)
        item = step["item"]
        print(f"[{i + 1}] {step['action']} {item} {step.get('grasp') or step.get('direction')} "
              f"risk={step.get('risk')} | {step.get('reason')}")
        print(f"     features: {_fmt(features(normalize(state), item))}")
        if step["action"] == "grasp":
            outcome = sim.grasp(item, step["grasp"])
        else:
            outcome = sim.push(item, step["direction"])
        after = _ids(sim.state())
        gone = sorted(set(before) - set(after))
        msg = f"     outcome={outcome}"
        if step["action"] == "grasp":
            msg += f"  removed={gone}"
        elif item in after:
            a, b = before[item]["pos"], after[item]["pos"]
            msg += f"  moved ({b[0] - a[0]:+.3f}, {b[1] - a[1]:+.3f}) z {a[2]:.3f}->{b[2]:.3f}"
        print(msg)


def main():
    path, policy = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "clearpick"
    from sim.sim import Sim
    decide = importlib.import_module(f"policies.{policy}").decide
    trace(json.load(open(path)), decide, Sim(width=64, height=48))


if __name__ == "__main__":
    main()
